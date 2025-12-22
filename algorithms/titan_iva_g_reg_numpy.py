import numpy as np
from time import time
from .initializations import _jbss_sos, _cca
from .algebra_toolbox_numpy import *
from .problem_simulation import *
import dask as dk

def cost_iva_g_reg(W,C,Rx,alpha):
    res = -np.sum(np.log(np.linalg.det(np.moveaxis(C,2,0))))/2
    res += 0.5*alpha*np.sum(np.trace((C-1)**2))
    res += np.trace(np.sum(np.einsum('kKn, nNK, KJNM, nMJ -> kJn',C,W,Rx,W),axis=2))/2
    res -= np.sum(np.log(np.abs(np.linalg.det(np.moveaxis(W,2,0)))))
    return res
# On ne traite pas le cas où une valeur singulière est inférieure à epsilon car ça ne peut
# pas arriver à cause du prox utilisé sauf à l'initialisation éventuellement mais on le néglige

def grad_H_W(W,C,Rx):
    return np.einsum('KJN,NMJ,JKMm->NmK',C,W,Rx)

def prox_f(W,c_w,mode='full'):
    if mode == 'full':
        U,s,Vh = np.linalg.svd(np.moveaxis(W,2,0))
        s_new = (s + np.sqrt(s**2 + 4*c_w))/2
        W_new = np.einsum('kNv,kvM -> kNM',U*s_new[:,np.newaxis],Vh)
        return np.moveaxis(W_new,0,2)
    elif mode =='blocks':
        for k,W_k in enumerate(W):
            for l,W_kl in enumerate(W_k):
                U,s,Vh = np.linalg.svd(W_kl)
                s_new = (s + np.sqrt(s**2 + 4*c_w))/2
                W_kl_new = U.dot(Vh*s_new)
                W[k][l] = W_kl_new
        return W

def prox_g(C,c_c,epsilon):
    s,U = np.linalg.eigh(np.moveaxis(C,2,0))
    Vh = np.moveaxis(U,1,2)
    s_new = np.maximum(epsilon,(s + np.sqrt(s**2+2*c_c))/2)
    # s_new = (s + np.sqrt(s**2+2*c_c))/2
    C_new = np.einsum('nNv,nvM -> nNM',U*s_new[:,np.newaxis],Vh)
    C_new = np.moveaxis(C_new,0,2)
    return sym_numpy(C_new)

def grad_H_C_reg(W,C,Rx,alpha):
    _,_,K = W.shape
    grad = sym_numpy(np.einsum('nNK, KJNM, nMJ -> KJn',W,Rx,W))/2
    grad[np.arange(K), np.arange(K), :] += alpha * (C[np.arange(K), np.arange(K), :] - 1) 
    return grad

def compute_L_W(C,Rx,nu,l_inf,boost):
    K,N = C.shape[0],C.shape[2]
    if boost:
        Hess = np.einsum('KJn,KJNM->nKNJM',C,Rx)
        Hess = np.reshape(Hess,(N,N*K,N*K))
        L_w = np.max(np.linalg.norm(Hess,ord=2,axis=(1,2)))
    else:
        L_w = lipschitz_numpy(C,spectral_norm_extracted_numpy(Rx,K,N))
    if nu > 0:
        L_w = max(l_inf,L_w)
    return L_w

#
# def palm_iva_g_reg(X,alpha=1,gamma_c=1.99,gamma_w=0.99,max_iter=5000,
#                    max_iter_int=100,crit_int = 1e-7,crit_ext = 1e-10,init_method='random',Winit=None,Cinit=None,
#                    eps=10**(-12),track_cost=False,
#                    seed=None,track_jisi=False,track_diff=False,B=None,adaptative_gamma_w=False,
#                    gamma_w_decay=0.9):
#     alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
#     N,T,K = X.shape
#     Lambda = np.einsum('NTK,MTJ->KJNM',X,X)/T
#     lam = spectral_norm_extracted_numpy(Lambda,K,N)
#     if (not adaptative_gamma_w) and gamma_w > 1:
#         raise('gamma_w must be in (0,1) if not adaptative')
#     if adaptative_gamma_w:
#         overhead = 0
#     W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Lambda,seed=seed)
#     N_step = 0
#     c_c = gamma_c/alpha
#     #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
#     diffs_W,diffs_C,jISI,cost = [],[],[],[]
#     if track_cost:
#         cost = [cost_iva_g_reg(W,C,Lambda,alpha)]
#     if track_jisi:
#         if np.any(B == None):
#             raise("you must provide B to track jISI")
#         else:
#             jISI = [joint_isi_numpy(W,B)]
#     diff_ext = np.inf
#     # N_updates_W, N_updates_C = update_scheme
#     while diff_ext > crit_ext and N_step < max_iter:
#         diff_int = np.inf
#         # for uw in range(N_updates_W):
#         N_step_int = 0
#         W_old0 = W.copy()
#         while diff_int > crit_int and N_step_int < max_iter_int:
#             W_old = W.copy()
#             c_w = gamma_w/max(eps,lipschitz_numpy(C,lam))
#             grad_W = grad_H_W(W,C,Lambda)
#             W = W - c_w*grad_W
#             W = prox_f(W,c_w)
#             diff_int = diff_criteria_numpy(W,W_old)
#             N_step_int += 1
#             # if diff_criteria(W,W_old) < W_diff_stop:
#             #     return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)
#         # print(N_step_int)
#         if track_cost:
#             cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
#         # if adaptative_gamma_w:
#         #     overhead -= time()
#         #     if cost_iva_g(W,C,Lambda) > cost_iva_g(W_old,C_old,Lambda):
#         #         gamma_w = gamma_w_decay*gamma_w
#         #     overhead += time()
#         # for uc in range(N_updates_C):
#         C_old = C.copy()
#         grad_C = grad_H_C_reg(W,C,Lambda,alpha)
#         C = C - c_c*grad_C
#         C = prox_g(C,c_c,eps)
#         diff_ext = max(diff_criteria_numpy(C,C_old),diff_criteria_numpy(W,W_old0))
#         if track_cost:
#             cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
#         # diff_W,diff_C = diff_criteria(W,W_old),diff_criteria(C,C_old)
#         # diff = max(diff_W,diff_C)
#         diff_W = diff_criteria_numpy(W,W_old)
#         if track_diff:
#             # diffs_W.append(diff_criteria(W,W_old))
#             diffs_W.append(diff_W)
#             # diffs_C.append(diff_C)
#         if track_jisi:
#             jISI.append(joint_isi_numpy(W,B))
#         N_step += 1
#     # if adaptative_gamma_w:
#     #     print(overhead)
#     return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)

def titan_iva_g_reg_numpy(X,alpha=1,gamma_c=1,gamma_w=0.99,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,crit_int=1e-10,crit_ext=1e-10,init_method='random',Winit=None,Cinit=None,inflate = False, lambda_inflate=1e-3,down_sample=False,num_samples=10000,epsilon=10**(-12),zeta=1e-3,track_times=True,track_costs=False,track_jisi=False,track_diffs=False,track_schemes=False,track_shifts=False,B=None,nu=0.5,adaptative_gamma_w=False,gamma_w_decay=0.9,boost=False,seed=None):
    X, alpha, gamma_c, gamma_w, N, K, Rx, rho_Rx = init_data_param(X, alpha, gamma_c, gamma_w, inflate, lambda_inflate, down_sample, num_samples)
    #Empiriquement, prend des valeurs entre 1 et 3 après whitening
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Rx,seed=seed)
    rho_bar = max(spectral_norm_numpy(C),3*K*(1+np.sqrt(1/(2*alpha*gamma_c))))
    l_sup = max((gamma_w*alpha)/(1-gamma_w),rho_Rx*rho_bar)
    C0 = min(4*gamma_c**2/(9*K**2),alpha*gamma_w/((1+zeta)*(1 - gamma_w)*l_sup))
    l_inf = (1+zeta)*C0*l_sup
    C_bar,C_tilde,C_prev,W_bar,W_tilde,W_prev=C.copy(),C.copy(),C.copy(),W.copy(),W.copy(),W.copy()
    c_c = gamma_c/alpha
    beta_c = np.sqrt(C0*nu*(1-nu))
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs,jISI,detailed_jISI,costs,detailed_costs,shifts_W,scheme,times,detailed_times = [],[],[],[],[],[],[],[],[]
    # shift_W = 1
    if track_costs:
        costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        if np.any(B == None):
            raise("you must provide B to track jISI")
        else:
            jISI.append(joint_isi_numpy(W,B))
    if track_times:
        t0 = time()
        times.append(0)
    diff_ext = np.inf
    N_step = 0
    L_w = compute_L_W(C,Rx,nu,l_inf,boost)
    # dW = np.zeros_like((W))
    # gamma_w0 = gamma_w
    while diff_ext > crit_ext and N_step < max_iter:
        W_old = W.copy()
        C_old = C.copy()
        L_w_prev = L_w
        L_w = compute_L_W(C,Rx,nu,l_inf,boost) #Empiriquement le module de Lipschitz semble prendre des valeurs entre 1 et 20 dans nos exemples
        c_w = gamma_w/L_w       
        diff_int_W = np.inf
        diff_int_C = np.inf
        N_step_int_W = 0
        N_step_int_C = 0         
        while diff_int_W > crit_int and N_step_int_W < max_iter_int_W:
            W, W_prev = update_W(gamma_w, nu, Rx, W, C, C0, W_prev, L_w, L_w_prev, c_w)
            diff_int_W = diff_criteria_numpy(W,W_prev)
            N_step_int_W += 1
            record_tracked_vars_detailed(alpha, track_times, track_costs, track_jisi, B, Rx, W, C, detailed_jISI, detailed_costs, detailed_times, t0)
            # dW_prev = dW.copy()
            # dW = W - W_prev
            # if np.any(dW_prev):
            #     shift_W = np.sum(dW*dW_prev)/(np.linalg.norm(dW)*np.linalg.norm(dW_prev))
            #     shifts_W.append(shift_W)
            # if adaptative_gamma_w:
                # if shift_W < 0:
                    # gamma_w = gamma_w*gamma_w_decay
                # if shift_W > 0.99:
                #     gamma_w = gamma_w/gamma_w_decay
        while diff_int_C > crit_int and N_step_int_C < max_iter_int_C:
            C, C_prev = update_C(alpha, epsilon, Rx, W, C, C_prev, c_c, beta_c)
            diff_int_C = diff_criteria_numpy(C,C_prev)
            N_step_int_C += 1
            record_tracked_vars_detailed(alpha, track_times, track_costs, track_jisi, B, Rx, W, C, detailed_jISI, detailed_costs, detailed_times, t0)
        diff_ext = record_tracked_vars(alpha, track_times, track_costs, track_jisi, track_diffs, track_schemes, B, Rx, W, C, diffs, jISI, costs, scheme, times, t0, W_old, C_old, N_step_int_W, N_step_int_C)
        N_step += 1
    return report_variables(W,C,N_step,max_iter,track_times,times,detailed_times,track_costs,costs,detailed_costs,track_jisi,jISI,detailed_jISI,track_diffs,diffs,track_schemes,scheme,track_shifts,shifts_W)

def record_tracked_vars(alpha, track_times, track_costs, track_jisi, track_diffs, track_schemes, B, Rx, W, C, diffs, jISI, costs, scheme, times, t0, W_old, C_old, N_step_int_W, N_step_int_C):
    if track_schemes:
        scheme.append([N_step_int_W,N_step_int_C])
    diff_W = diff_criteria_numpy(W,W_old)
    diff_C = diff_criteria_numpy(C,C_old)
    diff_ext = max(diff_W,diff_C)
    if track_diffs:
        diffs.append(diff_W,diff_C)
    if track_costs:
        costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        jISI.append(joint_isi_numpy(W,B))
    if track_times:
        times.append(time()-t0)
    return diff_ext

def update_C(alpha, epsilon, Rx, W, C, C_prev, c_c, beta_c):
    C_tilde = C + beta_c*(C-C_prev)
    grad_C = grad_H_C_reg(W,C_tilde,Rx,alpha)
    C_bar = C_tilde - c_c*grad_C
    C_prev = C.copy()
    C = prox_g(C_bar,c_c,epsilon)
    return C,C_prev

def update_W(gamma_w, nu, Rx, W, C, C0, W_prev, L_w, L_w_prev, c_w):
    beta_w = (1-gamma_w)*np.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
    W_tilde = W + beta_w*(W-W_prev)
    grad_W = grad_H_W(W_tilde,C,Rx)
    W_bar = W_tilde - c_w*grad_W
    W_prev = W.copy()
    W = prox_f(W_bar,c_w)
    return W,W_prev

def record_tracked_vars_detailed(alpha, track_times, track_costs, track_jisi, B, Rx, W, C, detailed_jISI, detailed_costs, detailed_times, t0):
    if track_costs:
        detailed_costs.append(cost_iva_g_reg(W,C,Rx,alpha))
    if track_jisi:
        detailed_jISI.append(joint_isi_numpy(W,B))
    if track_times:
        detailed_times.append(time()-t0)

def init_data_param(X, alpha, gamma_c, gamma_w, inflate, lambda_inflate, down_sample, num_samples):
    N,T,K = X.shape
    if down_sample and T > num_samples:
        T = num_samples
        X = X[:,:T, :]       
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    # if (not adaptative_gamma_w) and gamma_w > 1:
    #     raise('gamma_w must be in (0,1) if not adaptative')
    Rx = np.einsum('NTK,MTJ->KJNM',X,X)/T
    if inflate:
        for k in range(K):
            Rx[k,k,:,:] += lambda_inflate*np.eye(N)
    rho_Rx = spectral_norm_extracted_numpy(Rx,K,N)
    return X,alpha,gamma_c,gamma_w,N,K,Rx,rho_Rx

def block_titan_palm_iva_g_reg(X,idx_W,alpha=1,gamma_c=1,gamma_w=0.99,C0=0.999,nu=0.5,
                               max_iter=20000,max_iter_int=100,crit_int=1e-9,crit_ext=1e-9,
                               init_method='random',Winit=None,Cinit=None,eps=10**(-12),
                               track_cost=False,seed=None,track_jisi=False,track_diff=False,B=None):
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    N,T,K = X.shape
    Lambda = np.einsum('NTK,MTJ->KJNM',X,X)/T
    lam = spectral_norm_extracted_numpy(Lambda,K,N)
    W_full, C = initialize(N,K,init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Lambda,seed=seed)
    W_blocks = full_to_blocks(W_full,idx_W,K)
    C_lat,C_bar,C_tilde,C_prev = C.copy(),C.copy(),C.copy(),C.copy()
    W_lat,W_bar,W_tilde,W_prev = W_full.copy(),W_full.copy(),W_full.copy(),W_full.copy()
    N_step = 0
    if alpha == 0:
        c_c = gamma_c #Tout pas strictement positif est inférieur à une constante de Lipchitz du gradient selon C.
    else:
        c_c = gamma_c/alpha
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,jISI,cost = [],[],[],[]
    if track_cost:
        cost = [cost_iva_g_reg(W_full,C,Lambda,alpha)]
    if track_jisi:
        if np.any(B == None):
            raise("you must provide B to track jISI")
        else:
            jISI = [joint_isi_numpy(W_full,B)]
    diff_ext = np.inf
    L_w = lipschitz_numpy(C,lam)
    # N_updates_W, N_updates_C = update_scheme
    while diff_ext > crit_ext and N_step < max_iter:
        C_old = C.copy()
        L_w_prev = L_w
        L_w = lipschitz_numpy(C,lam)
        diff_int = np.inf
        W_old0 = W_blocks.copy()
        N_step_int = 0
        # for update in range(N_updates_W):
        while diff_int > crit_int and N_step_int < max_iter_int:
            W_old = W_blocks.copy()
            c_w = gamma_w/L_w
            beta_w = (1-gamma_w)*np.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
            tau_w = beta_w
            W_tilde = W_full + beta_w*(W_full-W_prev)
            W_bar = W_full + tau_w*(W_full-W_prev)
            grad_W = grad_H_W(W_bar,C,Lambda)
            W_lat = W_tilde - c_w*grad_W
            W_prev = W_full.copy()
            W_lat = full_to_blocks(W_lat)
            W_blocks = prox_f(W_lat,c_w,mode='blocks')
            diff_int = diff_criteria_numpy(W_blocks,W_old)
            W_full = blocks_to_full(W_blocks)
            N_step_int += 1
        # print(N_step_int)
        if track_cost:
            cost.append(cost_iva_g_reg(W_full,C,Lambda,alpha))
        # for update in range(N_updates_C):
        beta_c = np.sqrt(C0*nu*(1-nu))
        tau_c = beta_c
        C_tilde = C + beta_c*(C-C_prev)
        C_bar = C + tau_c*(C-C_prev)
        grad_C = grad_H_C_reg(W_full,C_bar,Lambda,alpha)
        C_lat = C_tilde - c_c*grad_C
        C_prev = C.copy()
        C = prox_g(C_lat,c_c,eps)
        diff_ext = max(diff_criteria_numpy(C,C_old),diff_criteria_numpy(W_blocks,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W_full,C,Lambda,alpha))
        # diff_W = diff_criteria(W,W_old)
        # diff_C = diff_criteria(C,C_old)
        # # diff = max(diff_W,diff_C)
        # if track_diff:
        #     diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_jisi:
            jISI.append(joint_isi_numpy(W_full,B))
        N_step += 1
    return report_variables(W_full,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)

def initialize(N,K,init_method,Winit=None,Cinit=None,X=None,Rx=None,seed=None):
    if Winit is not None and Cinit is not None:
        W,C = Winit.copy(),Cinit.copy()
    elif init_method == 'Jdiag':
        W,C = Jdiag_init(X,N,K,Rx)
    elif init_method == 'random':
        C = make_Sigma(K,N,rank=K+10,seed=seed)
        W = make_A(K,N,seed=seed)      
    return W,C

def Jdiag_init(X,K,Rx):
    if K > 2:
        # initialize with multi-set diagonalization (orthogonal solution)
        W = _jbss_sos(X, 0, 'whole')
    else:
        W = _cca(X)
    C = np.moveaxis(np.linalg.inv(np.einsum('nNK, KJNM, nMJ -> nKJ',W,Rx,W)),0,2)
    return W,C

def to_float64(alpha, gamma_c, gamma_w):
    alpha = np.float64(alpha) #On s'assure que tous les calculs soient réalisés en précision double
    gamma_w = np.float64(gamma_w)
    gamma_c = np.float64(gamma_c)
    return alpha,gamma_c,gamma_w

def report_variables(W, C, N_step, max_iter, track_times, times, detailed_times, 
                     track_costs, costs, detailed_costs, track_jisi, jISI, detailed_jISI, 
                     track_diffs, diffs, track_schemes, scheme, track_shifts, shifts_W):
    met_limit = (N_step < max_iter)
    results = {'W': W, 'C': C, 'met_limit': met_limit} 
    if track_times:
        results['times'] = np.array(times)
        results['detailed_times'] = np.array(detailed_times)
    if track_costs:
        results['costs'] = np.array(costs)
        results['detailed_costs'] = np.array(detailed_costs)
    if track_jisi:
        results['jisi'] = np.array(jISI)
        results['detailed_jisi'] = np.array(detailed_jISI)
    if track_diffs:
        results['diffs'] = np.array(diffs)
    if track_schemes:
        results['scheme'] = np.array(scheme)
    if track_shifts:
        results['shifts_W'] = np.array(shifts_W)  
    return results

            

