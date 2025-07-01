import torch
import numpy as np
from time import time
from .initializations import _jbss_sos, _cca
from .algebra_toolbox_torch import *
from .problem_simulation import *
import dask as dk

""" 
def cost_iva_g_reg(W, C, Rx, alpha):
    K, _, N = C.size()
    cost = -torch.sum(torch.log(torch.det(C.permute(2, 0, 1)))) / 2
    for k in range(K):
        cost += 0.5 * alpha * torch.sum((C[k, k, :] - 1) ** 2)
    W_bd = torch.block_diag(*[W[:, k, :] for k in range(K)])
    cost += torch.trace(torch.sum(torch.einsum('kKn, KNn, Ni, ijn -> kjn', C, W_bd, Rx, W_bd.t()), dim=2)) / 2
    cost -= torch.sum(torch.log(torch.abs(torch.det(W.permute(2, 0, 1)))))
    return cost
# On ne traite pas le cas où une valeur singulière est inférieure à epsilon car ça ne peut
# pas arriver à cause du prox utilisé sauf à l'initialisation éventuellement mais on le néglige
 """

def cost_iva_g_reg(W, C, Rx, alpha):
    det_C = torch.det(C.permute(2, 0, 1))  # Déterminant de C
    det_W = torch.det(W.permute(2, 0, 1))  # Déterminant de W
    tr_C = torch.trace((C - 1)**2)  # Trace de (C - 1)^2
    tr_term = torch.trace(torch.sum(torch.einsum('kKn, nNK, KJNM, nMJ -> kJn', (C, W, Rx, W)), dim=2)) / 2  # Terme de trace
    res = -torch.sum(torch.log(torch.abs(det_C))) / 2  # Premier terme
    res += 0.5 * alpha * tr_C  # Deuxième terme
    res += tr_term  # Troisième terme
    res -= torch.sum(torch.log(torch.abs(det_W)))  # Quatrième terme
    return res.item()  # Convertir le résultat en un scalaire Python


""" def grad_H_W(W, C, Rx):
    N, _, K = W.size()
    grad = torch.zeros((N, N, K), device=W.device)
    W_bd = block_diag(W)
    C = C.double()
    W_bd = W_bd.double()
    Rx = Rx.double()
    C = C.to(W_bd.device)
    Rx = Rx.to(W_bd.device)
    grad_tmp = torch.einsum('kKn, KNn, Ni -> kin',C,W_bd,Rx)

    for k in range(K):
        grad[:, :, k] = grad_tmp[k, k*N:(k+1)*N, :].t()
    return grad
 """


def grad_H_W(W, C, Rx):
    return torch.einsum('KJN,NMJ,JKMm->NmK',C,W,Rx)

def prox_f(W,c_w,mode='full'):
    if mode == 'full':
        N,_,K = W.size()
        W_perm = W.permute(2, 0, 1)
        U,s,Vh = torch.linalg.svd(W_perm)
        s_new = (s + torch.sqrt(s**2 + 4*c_w))/2
        diag_s = torch.diag_embed(s_new)   
        W_new = torch.einsum('kNv,kvw,kwM -> kNM',U,diag_s,Vh)
        return W_new.permute(1,2,0)   # same as np.moveaxis(W_new,0,2) in numpy ??
    
    elif mode =='blocks':
        for k,W_k in enumerate(W):
            for l,W_kl in enumerate(W_k):
                U,s,Vh = torch.svd(W_kl)
                s_new = (s + torch.sqrt(s**2 + 4*c_w))/2
                W_kl_new = torch.matmul(torch.matmul(U, torch.diag(s_new)), Vh)
                W[k][l] = W_kl_new
        return W

def prox_g(C,c_c,eps):
    C_perm = C.permute(2, 0, 1)
    s,U = torch.linalg.eigh(C_perm)
    Vh = U.permute(0, 2, 1)
    s_new = torch.maximum(torch.tensor(eps, device=C.device, dtype=C.dtype), (s + torch.sqrt(s**2 + 2 * c_c)) / 2)
    # s_new = (s + np.sqrt(s**2+2*c_c))/2
    diag_s = torch.diag_embed(s_new)  
    C_new = torch.einsum('nNv,nvw,nwM -> nNM',U,diag_s,Vh)
    C_new = C_new.permute(1, 2, 0)
    return sym_torch(C_new)

""" def grad_H_C_reg(W, C, Rx, alpha):
    device = W.device  # Get the device of tensor W
    W_bd = block_diag(W)
    #print('grad HC reg - Wbd device :',W_bd.device)
    W_bdT = W_bd.transpose(0, 1)
    W_bd = W_bd.float()
    Rx = Rx.float().to(device)  # Move Lambda to the same device
    W_bdT = W_bdT.float()
    grad = torch.einsum('kNn, Nv, vin-> kin', W_bd, Rx, W_bdT) / 2
    K, _, N = C.shape
    for k in range(K):
        grad[k, k, :] += alpha * (C[k, k, :] - 1)
    return sym(grad) """

def grad_H_C_reg(W, C, Rx, alpha):
    _, _,K = W.size()
    grad = sym_torch(torch.einsum('nNK,KJNM,nMJ->KJn',W,Rx,W)) / 2
    grad[torch.arange(K), torch.arange(K), :] += alpha * (C[torch.arange(K), torch.arange(K), :] - 1)
    return grad


def palm_iva_g_reg(X,alpha=1,gamma_c=1.99,gamma_w=0.99,max_iter=5000,
                   max_iter_int=100,crit_int = 1e-7,crit_ext = 1e-10,init_method='random',Winit=None,Cinit=None,
                   eps=10**(-12),track_cost=False,
                   seed=None,track_isi=False,track_diff=False,B=None,adaptative_gamma_w=False,
                   gamma_w_decay=0.9):
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    N,T,K = X.size()
    Lambda = torch.einsum('NTK,MTJ->KJNM', X, X) / T
    lam = spectral_norm_extracted_torch(Lambda,K,N)
    if (not adaptative_gamma_w) and gamma_w > 1:
        raise('gamma_w must be in ]0,1[ if not adaptative')
    if adaptative_gamma_w:
        overhead = 0
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Lambda=Lambda,seed=seed)
    N_step = 0
    c_c = gamma_c/alpha
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,ISI,cost = [],[],[],[]
    if track_cost:
        cost = [cost_iva_g_reg(W,C,Lambda,alpha)]
    if track_isi:
        if np.any(B == None):
            raise("you must provide B to track ISI")
        else:
            ISI = [joint_isi_torch(W,B)]
    diff_ext = np.infty
    # N_updates_W, N_updates_C = update_scheme
    while diff_ext > crit_ext and N_step < max_iter:
        diff_int = np.infty
        # for uw in range(N_updates_W):
        N_step_int = 0
        W_old0 = W.clone()
        while diff_int > crit_int and N_step_int < max_iter_int:
            W_old = W.clone()
            c_w = gamma_w/max(eps,lipschitz(C,lam))
            grad_W = grad_H_W(W,C,Lambda)
            W = W - c_w*grad_W
            W = prox_f(W,c_w)
            diff_int = diff_criteria_torch(W,W_old)
            N_step_int += 1
            # if diff_criteria(W,W_old) < W_diff_stop:
            #     return report_variables(W,C,N_step,max_iter,cost,ISI,diffs_W,diffs_C,track_diff,track_cost,track_isi)
        # print(N_step_int)
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
        # if adaptative_gamma_w:
        #     overhead -= time()
        #     if cost_iva_g(W,C,Lambda) > cost_iva_g(W_old,C_old,Lambda):
        #         gamma_w = gamma_w_decay*gamma_w
        #     overhead += time()
        # for uc in range(N_updates_C):
        C_old = C.clone()
        grad_C = grad_H_C_reg(W,C,Lambda,alpha)
        C = C - c_c*grad_C
        C = prox_g(C,c_c,eps)
        diff_ext = max(diff_criteria_torch(C,C_old),diff_criteria_torch(W,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
        # diff_W,diff_C = diff_criteria(W,W_old),diff_criteria(C,C_old)
        # diff = max(diff_W,diff_C)
        diff_W = diff_criteria_torch(W,W_old)
        if track_diff:
            # diffs_W.append(diff_criteria(W,W_old))
            diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_isi:
            ISI.append(joint_isi_torch(W,B))
        N_step += 1
    # if adaptative_gamma_w:
    #     print(overhead)
    return report_variables(W,C,N_step,max_iter,cost,ISI,diffs_W,diffs_C,track_diff,track_cost,track_isi)

def titan_iva_g_reg_torch(X,alpha=1,gamma_c=1,gamma_w=0.99,max_iter=20000,
                         max_iter_int=100,crit_int=1e-10,crit_ext=1e-10,init_method='random',
                         Winit=None,Cinit=None,
                         eps=10**(-12),track_cost=False,seed=None,
                         track_jisi=False,track_diff=False,B=None,nu=0.5,zeta = 1e-3,
                         max_iter_int_C=1,adaptative_gamma_w=False,
                         gamma_w_decay=0.9,boost=False):
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
    if Winit is not None:
        Winit = torch.tensor(Winit)
    if Cinit is not None:
        Cinit = torch.tensor(Cinit)
    if B is not None:
        B = torch.tensor(B)
    alpha, gamma_c, gamma_w = torch.tensor(alpha), torch.tensor(gamma_c), torch.tensor(gamma_w)

    # if (not adaptative_gamma_w) and gamma_w > 1:
    #     raise('gamma_w must be in (0,1) if not adaptative')
    N,T,K = X.size()
    Rx = torch.einsum('NTK,MTJ->KJNM',X,X)/T
    rho_Rx = spectral_norm_extracted_torch(Rx,K,N)
    W,C = initialize(N,K,init_method=init_method,Winit=Winit,Cinit=Cinit,X=X,Rx=Rx,seed=seed)
    l_sup = torch.max((gamma_w*alpha)/(1-gamma_w),rho_Rx*2*K*(1+torch.sqrt(2/(alpha*gamma_c))))
    print(X.dtype,C.dtype,W.dtype,Rx.dtype)
    # Utiliser torch.minimum pour trouver le minimum entre deux tenseurs à la fois
    C0_tmp = torch.minimum(gamma_c**2 / K**2, alpha * gamma_w / ((1 + zeta) * (1 - gamma_w) * l_sup))
    C0 = torch.minimum(C0_tmp, rho_Rx / ((1 + zeta) * l_sup))
    # C0 = torch.min(gamma_c**2/K**2,alpha*gamma_w/((1+zeta)*(1 - gamma_w)*l_sup),rho_Rx/((1+zeta)*l_sup))
    l_inf = (1+zeta)*C0*l_sup
    C_lat,C_bar,C_tilde,C_prev,W_lat,W_bar,W_tilde,W_prev = C.clone(),C.clone(),C.clone(),C.clone(),W.clone(),W.clone(),W.clone(),W.clone()
    N_step = 0
    c_c = gamma_c/alpha
    beta_c = torch.sqrt(C0*nu*(1-nu))
    tau_c = beta_c
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,ISI,cost,shifts_W = [],[],[],[],[]
    # shift_W = 1
    if track_cost:
        cost = [cost_iva_g_reg(W,C,Rx,alpha)]
    if track_jisi:
        if np.any(B == None):
            raise("you must provide B to track jISI")
        else:
            jISI = [joint_isi_torch(W,B)]
    else:
        jISI = []
    diff_ext = np.inf
    L_w = torch.max(l_inf,lipschitz(C,rho_Rx))
    #print('L_w device :',L_w.device)
    times = [0]
    t0 = time()
    # dW = np.zeros_like((W))
    # gamma_w0 = gamma_w
    while diff_ext > crit_ext and N_step < max_iter:
        C_old0 = C.clone()
        L_w_prev = L_w
        L_w = torch.max(l_inf,lipschitz(C,rho_Rx))
        diff_int = np.inf
        diff_int_C = np.inf #here
        W_old0 = W.clone()
        N_step_int_W = 0
        N_step_int_C = 0 #here
        c_w = gamma_w/L_w
        # gamma_w = gamma_w0
        while diff_int > crit_int and N_step_int_W < max_iter_int:
            # W_old = W.clone()
            beta_w = (1-gamma_w)*torch.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
            tau_w = beta_w
            W_tilde = W + beta_w*(W-W_prev)
            W_bar = W + tau_w*(W-W_prev)
            #print('W_bar dimensions :',W_bar.dim())
            grad_W = grad_H_W(W_bar,C,Rx)
            #print('grad_W :',grad_W.device)
            W_lat = W_tilde - c_w*grad_W
            W_prev = W.clone()
            W = prox_f(W_lat,c_w)
            diff_int = diff_criteria_torch(W,W_prev)
            N_step_int_W += 1
            # dW_prev = dW.clone()
            # dW = W - W_prev
            # if np.any(dW_prev):
            #     shift_W = np.sum(dW*dW_prev)/(np.linalg.norm(dW)*np.linalg.norm(dW_prev))
            #     shifts_W.append(shift_W)
            # if adaptative_gamma_w:
                # if cost_iva_g_reg(W,C,Lambda,alpha) > cost_iva_g_reg(W_prev,C,Lambda,alpha):
                # if shift_W < 0:
                    # print(cost_iva_g_reg(W,C,Lambda,alpha))
                    # gamma_w = gamma_w*gamma_w_decay
                # if shift_W > 0.99:
                #     gamma_w = gamma_w/gamma_w_decay
            # if diff_criteria(W,W_old) < W_diff_stop:
            #     return report_variables(W,C,N_step,max_iter,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi)    
        # print(N_step_int)
        # if track_cost:
        #     cost.append(cost_iva_g_reg(W,C,Lambda,alpha))
        while diff_int_C > crit_int and N_step_int_C < max_iter_int_C: #here
            # C_old = C.clone() #here
        
            C_tilde = C + beta_c*(C-C_prev)
            C_bar = C + tau_c*(C-C_prev)
            grad_C = grad_H_C_reg(W,C_bar,Rx,alpha)
            C_lat = C_tilde - c_c*grad_C
            C_prev = C.clone()
            C = prox_g(C_lat,c_c,eps)
            diff_int = diff_criteria_torch(C,C_prev) #here
            N_step_int_C += 1 #here
        diff_ext = max(diff_criteria_torch(C,C_old0),diff_criteria_torch(W,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W,C,Rx,alpha))
        # diff_W = diff_criteria(W,W_old)
        # diff_C = diff_criteria(C,C_old)
        # # diff = max(diff_W,diff_C)
        # if track_diff:
        #     diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_jisi:
            jISI.append(joint_isi_torch(W,B))
        times.append(time()-t0)
        N_step += 1
    return report_variables(W,C,N_step,max_iter,times,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_jisi,shifts_W)


def block_titan_palm_iva_g_reg(X,idx_W,alpha=1,gamma_c=1,gamma_w=0.99,C0=0.999,nu=0.5,
                               max_iter=20000,max_iter_int=100,crit_int=1e-9,crit_ext=1e-9,
                               init_method='random',Winit=None,Cinit=None,eps=10**(-12),
                               track_cost=False,seed=None,track_isi=False,track_diff=False,B=None):
    alpha, gamma_c, gamma_w = to_float64(alpha, gamma_c, gamma_w)
    N,T,K = X.size()
    Lambda = torch.einsum('NTK,MTJ->KJNM', X, X) / T
    lam = spectral_norm_extracted_torch(Lambda,K,N)
    W_full, C = initialize(N,K,init_method,Winit=Winit,Cinit=Cinit,X=X,Lambda=Lambda,seed=seed)
    W_blocks = full_to_blocks(W_full,idx_W,K)
    C_lat,C_bar,C_tilde,C_prev = C.clone(),C.clone(),C.clone(),C.clone()
    W_lat,W_bar,W_tilde,W_prev = W_full.clone(),W_full.clone(),W_full.clone(),W_full.clone()
    N_step = 0
    if alpha == 0:
        c_c = gamma_c #Tout pas strictement positif est inférieur à une constante de Lipchitz du gradient selon C.
    else:
        c_c = gamma_c/alpha
    #On initialise les listes utiles pour tracer les courbes. Par défaut on garde les critères pour les deux blocs, on pourra calculer le max a posteriori si besoin
    diffs_W,diffs_C,ISI,cost = [],[],[],[]
    if track_cost:
        cost = [cost_iva_g_reg(W_full,C,Lambda,alpha)]
    if track_isi:
        if np.any(B == None):
            raise("you must provide B to track ISI")
        else:
            ISI = [joint_isi_torch(W_full,B)]
    diff_ext = np.inf
    L_w = lipschitz(C,lam)
    # N_updates_W, N_updates_C = update_scheme
    while diff_ext > crit_ext and N_step < max_iter:
        C_old = C.clone()
        L_w_prev = L_w
        L_w = lipschitz(C,lam)
        diff_int = np.inf
        W_old0 = W_blocks.clone()
        N_step_int = 0
        # for update in range(N_updates_W):
        while diff_int > crit_int and N_step_int < max_iter_int:
            W_old = W_blocks.clone()
            c_w = gamma_w/L_w
            beta_w = (1-gamma_w)*np.sqrt(C0*nu*(1-nu)*L_w_prev/L_w)
            tau_w = beta_w
            W_tilde = W_full + beta_w*(W_full-W_prev)
            W_bar = W_full + tau_w*(W_full-W_prev)
            grad_W = grad_H_W(W_bar,C,Lambda)
            W_lat = W_tilde - c_w*grad_W
            W_prev = W_full.clone()
            W_lat = full_to_blocks(W_lat)
            W_blocks = prox_f(W_lat,c_w,mode='blocks')
            diff_int = diff_criteria_torch(W_blocks,W_old)
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
        C_prev = C.clone()
        C = prox_g(C_lat,c_c,eps)
        diff_ext = max(diff_criteria_torch(C,C_old),diff_criteria_torch(W_blocks,W_old0))
        if track_cost:
            cost.append(cost_iva_g_reg(W_full,C,Lambda,alpha))
        # diff_W = diff_criteria(W,W_old)
        # diff_C = diff_criteria(C,C_old)
        # # diff = max(diff_W,diff_C)
        # if track_diff:
        #     diffs_W.append(diff_W)
            # diffs_C.append(diff_C)
        if track_isi:
            ISI.append(joint_isi_torch(W_full,B))
        N_step += 1
    return report_variables(W_full,C,N_step,max_iter,cost,ISI,diffs_W,diffs_C,track_diff,track_cost,track_isi)

def initialize(N,K,init_method,Winit=None,Cinit=None,X=None,Rx=None,seed=None):
    if Winit is not None and Cinit is not None:
        W,C = Winit.clone(),Cinit.clone()
    elif init_method == 'Jdiag':
        W,C = Jdiag_init(X,N,K,Rx)
    elif init_method == 'random':
        C = make_Sigma(K,N,rank=K+10,seed=seed)
        W = make_A(K,N,seed=seed)        
    return torch.from_numpy(W).float(),torch.from_numpy(C).float()

def Jdiag_init(X,N,K,Rx):
    if K > 2:
        # initialize with multi-set diagonalization (orthogonal solution)
        W = _jbss_sos(X, 0, 'whole')
    else:
        W = _cca(X)
    C = torch.moveaxis(torch.linalg.inv(torch.einsum('nNK, KJNM, nMJ -> nKJ',W,Rx,W)),0,2)
    return W,C

def to_float64(alpha, gamma_c, gamma_w):
    alpha = np.float64(alpha) #On s'assure que tous les calculs soient réalisés en précision double
    gamma_w = np.float64(gamma_w)
    gamma_c = np.float64(gamma_c)
    return alpha,gamma_c,gamma_w

def report_variables(W,C,N_step,max_iter,times,cost,jISI,diffs_W,diffs_C,track_diff,track_cost,track_isi,shifts_W):
    if N_step < max_iter:
        met_limit = True
    else:
        met_limit = False
    if track_diff:
        # diffs = (diffs_W,diffs_C)
        diffs = diffs_W
        if track_cost:
            if track_isi:
                return W,C,met_limit,times,cost,jISI,diffs
            else:
                return W,C,met_limit,times,cost,diffs
        else:
            if track_isi:
                return W,C,met_limit,times,jISI,diffs
            else:
                return W,C,met_limit,times,diffs
    else:
        if track_cost:
            if track_isi:
                return W,C,met_limit,times,cost,jISI
            else:
                return W,C,met_limit,times,cost
        else:
            if track_isi:
                return W,C,met_limit,times,jISI #,shifts_W
            else:
                return W,C,met_limit,times
            

