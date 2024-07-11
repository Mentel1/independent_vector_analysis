import torch
import numpy as np
from tools import *
from data import *
from initializations import _jbss_sos, _cca



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



def grad_H_W(W, C, Rx):
    return torch.einsum('bKJN,bNMJ,bJKMm->bNmK',C,W,Rx)



""" # proxf without batch
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
        return W """
    


def prox_f(W, c_w, mode='full'):
    if mode == 'full':
        B, N, _, K = W.size()  # Assume W has shape [B, N, _, K]
        W_perm = W.permute(0, 3, 1, 2)  # Permute to shape [K, B, N, _]
        U, s, Vh = torch.linalg.svd(W_perm)
        c_w = c_w.view(-1, 1, 1)  # Reshape c_w to broadcast correctly
        s_new = (s + torch.sqrt(s**2 + 4*c_w)) / 2
        diag_s = torch.diag_embed(s_new)
        W_new = torch.einsum('bknv,bkvw,bkwm -> bknm', U, diag_s, Vh)
        return W_new.permute(0,2,3,1)  # Shape back to [B, N, _, K]
    
    elif mode == 'blocks':
        B, N, _, K = W.size()
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    W_bnk = W[b, n, :, k]
                    U, s, Vh = torch.svd(W_bnk)
                    s_new = (s + torch.sqrt(s**2 + 4*c_w)) / 2
                    W_bnk_new = torch.matmul(torch.matmul(U, torch.diag(s_new)), Vh)
                    W[b, n, :, k] = W_bnk_new
        return W




""" # proxg without batch
def prox_g(C,c_c,eps):
    C_perm = C.permute(2, 0, 1)
    s,U = torch.linalg.eigh(C_perm)
    Vh = U.permute(0, 2, 1)
    s_new = torch.maximum(torch.tensor(eps, device=C.device, dtype=C.dtype), (s + torch.sqrt(s**2 + 2 * c_c)) / 2)
    # s_new = (s + np.sqrt(s**2+2*c_c))/2
    diag_s = torch.diag_embed(s_new)  
    C_new = torch.einsum('nNv,nvw,nwM -> nNM',U,diag_s,Vh)
    #C_new = C_new.permute(1, 2, 0)
    return sym(C_new) """



def prox_g(C, c_c, eps):
    # Permute the tensor to get the eigenvalues in the correct shape
    C_perm = C.permute(0, 3, 1, 2)  # (batch, last_dim, height, width)
    s, U = torch.linalg.eigh(C_perm)
    Vh = U.transpose(-1, -2)
    # Calculate the new singular values
    #print("s size: ", s.size()) 
    #print("c_c size: ", c_c.size())
    c_c = c_c.view(-1, 1, 1)  # Reshape c_c to broadcast correctly
    s_new = torch.maximum(torch.tensor(eps, device=C.device, dtype=C.dtype), (s + torch.sqrt(s**2 + 2 * c_c)) / 2)
    diag_s = torch.diag_embed(s_new)
    
    # Reconstruct the matrix with the new singular values
    C_new = torch.einsum('bnNv,bnvw,bnwM -> bnNM',U,diag_s,Vh)
    C_new = C_new.permute(0, 2, 3, 1)  # Permute back to original shape
    return sym(C_new)


def grad_H_C_reg(W, C, Rx, alpha):
    B, N, _, K = W.size()
    
    # Compute the gradient with respect to H
    grad = sym(torch.einsum('bnNK,bKJNM,bnMJ->bKJn', W, Rx, W)) / 2
    
    # Add the regularization term
    indices = torch.arange(K).to(W.device)
    #print("C size: ", C.size())
    alpha = alpha.view(B, 1, 1)
    #print("alpha size: ", alpha.size())
    grad[:, indices, indices, :] += alpha * (C[:, indices, indices, :] - 1)
    
    return grad


## a changer pour batch
def Jdiag_init(X,N,K,Rx):
    if K > 2:
        # initialize with multi-set diagonalization (orthogonal solution)
        W = _jbss_sos(X, 0, 'whole')
    else:
        W = _cca(X)
    W_bd = block_diag(W)
    W_bdT = np.moveaxis(W_bd,0,1)
    Sigma_tmp = np.einsum('KNn, Ni, ijn -> Kjn',W_bd,Rx,W_bdT)
    C = np.zeros((K,K,N))
    for n in range(N):
        C[:,:,n] = np.linalg.inv(Sigma_tmp[:,:,n])
    return W,C


def initialize(N,K,init_method='random',Winit=None,Cinit=None,X=None,Rx=None,seed=None):
    if Winit is not None and Cinit is not None:
        W,C = Winit.clone(),Cinit.clone()
    elif init_method == 'Jdiag':
        W,C = Jdiag_init(X,N,K,Rx)
    elif init_method == 'random':
        C = make_Sigma(K,N,rank=K+10,seed=seed)
        W = make_A(K,N,seed=seed)      
    W = W.cuda()
    C = C.cuda()  
    return W,C




