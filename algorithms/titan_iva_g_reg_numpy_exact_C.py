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

def prox_g(C,c_c,eps):
    s,U = np.linalg.eigh(np.moveaxis(C,2,0))
    Vh = np.moveaxis(U,1,2)
    s_new = np.maximum(eps,(s + np.sqrt(s**2+2*c_c))/2)
    # s_new = (s + np.sqrt(s**2+2*c_c))/2
    C_new = np.einsum('nNv,nvM -> nNM',U*s_new[:,np.newaxis],Vh)
    C_new = np.moveaxis(C_new,0,2)
    return sym_numpy(C_new)

def grad_H_C_reg(W,C,Rx,alpha):
    _,_,K = W.shape
    grad = sym_numpy(np.einsum('nNK, KJNM, nMJ -> KJn',W,Rx,W))/2
    grad[np.arange(K), np.arange(K), :] += alpha * (C[np.arange(K), np.arange(K), :] - 1) 
    return grad

def palm_iva_g_reg(X,alpha=1,gamma_c=1.99,gamma_w=0.99,max_iter=5000,
                   max_iter_int=100,crit_int = 1e-7,crit_ext = 1e-10,init_method='random',Winit=None,Cinit=None,
                   eps=10**(-12),track_cost=False,
                   seed=None,track_jisi=False,track_diff=False,B=None,adaptative_gamma_w=False,
                   gamma_w_decay=0.9):
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    N,T,K = X.shape
    Lambda = np.einsum('NTK,MTJ->KJNM',X,X)/T
    lam = spectral_norm_extracted_numpy(Lambda,K,N)
    if (not adaptative_gamma_w) and gamma_w > 1:
        raise('gamma_w must be in (0,1) if not adaptative')
    if adaptative_gamma_w:
        overhead = 0
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Lambda,seed=seed)
    N_step = 0
    c_c = gamma_c/alpha
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,jISI,cost = [],[],[],[]
    if track_cost:
        cost = [cost_iva_g_reg(W,C,Lambda,alpha)]
    if track_jisi:
        if np.any(B == None):
            raise("you must provide B to track jISI")
        else:
            jISI = [joint_isi_numpy(W,B)]
    diff_ext = np.inf
    # N_updates_W, N_updates_C = update_scheme
    while diff_ext > crit_ext and N_step < max_iter:
        diff_int = np.inf
        # for uw in range(N_updates_W):
        N_step_int = 0
        W_old0 = W.copy()
        while diff_int > crit_int and N_step_int < max_iter_int:
            W_old = W.copy()
            c_w = gamma_w/max(eps,lipschitz_numpy(C,lam))
            grad_W = grad_H_W(W,C,Lambda)
            W = W - c_w*grad_W
            W = prox_f(W,c_w)
            diff_int = diff_criteria_numpy(W,W_old)
            N_step_int += 1
            # if diff_criteria(W,W_old) < W_diff_stop:
            #     return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)
        # print(N_step_int)
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
        # if adaptative_gamma_w:
        #     overhead -= time()
        #     if cost_iva_g(W,C,Lambda) > cost_iva_g(W_old,C_old,Lambda):
        #         gamma_w = gamma_w_decay*gamma_w
        #     overhead += time()
        # for uc in range(N_updates_C):
        C_old = C.copy()
        grad_C = grad_H_C_reg(W,C,Lambda,alpha)
        C = C - c_c*grad_C
        C = prox_g(C,c_c,eps)
        diff_ext = max(diff_criteria_numpy(C,C_old),diff_criteria_numpy(W,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
        # diff_W,diff_C = diff_criteria(W,W_old),diff_criteria(C,C_old)
        # diff = max(diff_W,diff_C)
        diff_W = diff_criteria_numpy(W,W_old)
        if track_diff:
            # diffs_W.append(diff_criteria(W,W_old))
            diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_jisi:
            jISI.append(joint_isi_numpy(W,B))
        N_step += 1
    # if adaptative_gamma_w:
    #     print(overhead)
    return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)

def titan_iva_g_reg_numpy_exactC(X,alpha=1,gamma_c=1,gamma_w=0.99,max_iter=20000,
                         max_iter_int=100,crit_int=1e-10,crit_ext=1e-10,init_method='random',
                         Winit=None,Cinit=None,
                         eps=10**(-12),track_cost=False,seed=None,
                         track_jisi=False,track_diff=False,B=None,nu=0.5,zeta=1e-3,
                         max_iter_int_C=1,adaptative_gamma_w=False,
                         gamma_w_decay=0.9,boost=False):
    N,T,K = X.shape
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    # if (not adaptative_gamma_w) and gamma_w > 1:
    #     raise('gamma_w must be in (0,1) if not adaptative')
    Rx = np.einsum('NTK,MTJ->KJNM',X,X)/T
    rho_Rx = spectral_norm_extracted_numpy(Rx,K,N)
    #Empiriquement, prend des valeurs entre 1 et 3 après whitening
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Rx,seed=seed)
    l_sup = max((gamma_w*alpha)/(1-gamma_w),rho_Rx*2*K*(1+np.sqrt(2/(alpha*gamma_c))))
    C0 = min(gamma_c**2/K**2,alpha*gamma_w/((1+zeta)*(1 - gamma_w)*l_sup),rho_Rx/((1+zeta)*l_sup))
    l_inf = (1+zeta)*C0*l_sup
    C_lat,C_bar,C_tilde,C_prev,W_lat,W_bar,W_tilde,W_prev = C.copy(),C.copy(),C.copy(),C.copy(),W.copy(),W.copy(),W.copy(),W.copy()
    N_step = 0
    c_c = gamma_c/alpha
    beta_c = np.sqrt(C0*nu*(1-nu))
    tau_c = beta_c
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,jISI,cost,shifts_W = [],[],[],[],[]
    # shift_W = 1
    if track_cost:
        cost = [cost_iva_g_reg(W,C,Rx,alpha)]
    if track_jisi:
        if np.any(B == None):
            raise("you must provide B to track jISI")
        else:
            jISI = [joint_isi_numpy(W,B)]
    else:
        jISI = []
    diff_ext = np.inf
    L_w = max(l_inf,lipschitz_numpy(C,rho_Rx))
    times = [0]
    t0 = time()
    # dW = np.zeros_like((W))
    # gamma_w0 = gamma_w
    while diff_ext > crit_ext and N_step < max_iter:
        C_old0 = C.copy()
        L_w_prev = L_w
        L_w = max(l_inf,lipschitz_numpy(C,rho_Rx)) #Empiriquement le module de Lipschitz semble prendre des valeurs entre 1 et 20 dans nos exemples       
        diff_int = np.inf
        diff_int_C = np.inf #here
        W_old0 = W.copy()
        N_step_int_W = 0
        N_step_int_C = 0 #here
        c_w = gamma_w/L_w
        # if boost:
        #     Hess = np.einsum('KJn,KJNM->nKNJM',C,Rx)
        #     Hess = np.reshape(Hess,(N,N*K,N*K))
        #     c_w = gamma_w/np.max(np.linalg.norm(Hess,ord=2,axis=(1,2)))
        # gamma_w = gamma_w0
        while diff_int > crit_int and N_step_int_W < max_iter_int:
            # W_old = W.copy() 
            beta_w = (1-gamma_w)*np.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
            tau_w = beta_w
            W_tilde = W + beta_w*(W-W_prev)
            W_bar = W + tau_w*(W-W_prev)
            grad_W = grad_H_W(W_bar,C,Rx)
            W_lat = W_tilde - c_w*grad_W
            W_prev = W.copy()
            W = prox_f(W_lat,c_w)
            diff_int = diff_criteria_numpy(W,W_prev)
            N_step_int_W += 1
            # dW_prev = dW.copy()
            # dW = W - W_prev
            # if np.any(dW_prev):
            #     shift_W = np.sum(dW*dW_prev)/(np.linalg.norm(dW)*np.linalg.norm(dW_prev))
            #     shifts_W.append(shift_W)
            # if adaptative_gamma_w:
                # if cost_iva_g_reg(W,C,Rx,alpha) > cost_iva_g_reg(W_prev,C,Rx,alpha):
                # if shift_W < 0:
                    # print(cost_iva_g_reg(W,C,Rx,alpha))
                    # gamma_w = gamma_w*gamma_w_decay
                # if shift_W > 0.99:
                #     gamma_w = gamma_w/gamma_w_decay
            # if diff_criteria(W,W_old) < W_diff_stop:
            #     return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)    
        # print(N_step_int)
        # if track_cost:
        #     cost.append(cost_iva_g_reg(W,C,Rx,alpha))
        # while diff_int_C > crit_int and N_step_int_C < max_iter_int_C: #here
        #     # C_old = C.copy() #here
        #     C_tilde = C + beta_c*(C-C_prev)
        #     C_bar = C + tau_c*(C-C_prev)
        #     grad_C = grad_H_C_reg(W,C_bar,Rx,alpha)
        #     C_lat = C_tilde - c_c*grad_C
        #     C_prev = C.copy()
        #     C = prox_g(C_lat,c_c,eps)
        #     diff_int = diff_criteria_numpy(C,C_prev) #here
        #     N_step_int_C += 1 #here
        C = np.linalg.inv(np.einsum('nNK, KJNM, nMJ -> nKJ',W,Rx,W))
        C = np.moveaxis(C,0,2)
        diff_ext = max(diff_criteria_numpy(C,C_old0),diff_criteria_numpy(W,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Rx,alpha))
        # diff_W = diff_criteria(W,W_old)
        # diff_C = diff_criteria(C,C_old)
        # # diff = max(diff_W,diff_C)
        # if track_diff:
        #     diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_jisi:
            jISI.append(joint_isi_numpy(W,B))
        times.append(time()-t0)
        N_step += 1
    return report_variables(W,C,N_step,max_iter,times,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi,shifts_W)

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

def Jdiag_init(X,N,K,Rx):
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

def report_variables(W,C,N_step,max_iter,times,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi,shifts_W):
    met_limit = (N_step < max_iter)
    if track_diff:
        # diffs = (diffs_W,diffs_C)
        diffs = diffs_W
        if track_cost:
            if track_jisi:
                return W,C,met_limit,times,cost,jISI,diffs
            else:
                return W,C,met_limit,times,cost,diffs
        else:
            if track_jisi:
                return W,C,met_limit,times,jISI,diffs
            else:
                return W,C,met_limit,times,diffs
    else:
        if track_cost:
            if track_jisi:
                return W,C,met_limit,times,cost,jISI
            else:
                return W,C,met_limit,times,cost
        else:
            if track_jisi:
                return W,C,met_limit,times,jISI #,shifts_W
            else:
                return W,C,met_limit,times
            

