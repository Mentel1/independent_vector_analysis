import torch
import numpy as np
from tools import *
from torch.utils.data import Dataset
from helpers_iva import whiten_data 
import os

 
class MyDatasetFolder(Dataset):
    def __init__(self, base_path, K, N, size):
        self.path = os.path.join(base_path, f"K_{K}_N_{N}")
        self.all_files = [f for f in os.listdir(self.path) if f.endswith(".pt")]
        #print("files in path: ", self.all_files)
        
        # Take only the first `size` files
        self.data_files = self.all_files[:size]
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        
        X_A_file = os.path.join(self.path, self.data_files[idx])
        #print("X_A_file: ", X_A_file)
        X_A_item = torch.load(X_A_file)
        #print(f"X_A_item: {X_A_item}")

        X_item, A_item = X_A_item[0], X_A_item[1]
        return X_item, A_item







## Problem simumation functions 


def make_A(K,N,seed=None):
    if seed == None:
        A = torch.randn(N,N,K)
    else:
        torch.manual_seed(seed)
        A = torch.randn(N, N, K)
    A = A.to(device='cuda')
    return A


def make_A_debug(K, N, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    A = torch.randn(N, N, K)
    
    # Debugging prints
    print(f"A shape: {A.shape}")
    print(f"A min value: {A.min()}, A max value: {A.max()}")
    
    try:
        #A = A.to(device='cuda')
        pass
    except RuntimeError as e:
        print(f"Error when moving to CUDA: {e}")
        return None
    
    return A



def make_Sigma(K,N,rank,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25,seed=None,normalize=False):
    
    rng = np.random.default_rng(seed)
    #if seed is not None :
    #    torch.manual_seed(seed)
    
    J = torch.ones(K, K)
    I = torch.eye(K)
    Q = torch.zeros(K, rank, N)
    mean = torch.zeros(K)
    Sigma = torch.zeros(K, K, N)
    if N == 1:
        rho = [torch.mean(rho_bounds)]
    else:
        rho = [(n/(N-1))*rho_bounds[1] + (1-(n/(N-1)))*rho_bounds[0] for n in range(N)]
    for n in range(N):
        eta = 1 - lambda_ - rho[n]
        if eta < 0 or lambda_ < 0 or rho[n] < 0:
            raise("all three coefficients must belong to [0,1]") 
        Q[:,:,n] = torch.tensor(rng.multivariate_normal(mean,I,rank).T)
        #Q[:,:,n] = torch.distributions.multivariate_normal.MultivariateNormal(mean,I).sample((rank,rank)).T
        if normalize:
            Q[:, :, n] = (Q[:, :, n].t() / torch.norm(Q[:, :, n], dim=1)).t()
            Sigma[:,:,n] = rho[n]*J + eta*I + lambda_*torch.matmul(Q[:, :, n], Q[:, :, n].t())
        else:
            Sigma[:,:,n] = rho[n]*J + eta*I + (lambda_/rank)*torch.matmul(Q[:, :, n], Q[:, :, n].t())
    for n in range(1,N):
        Sigma[:,:,n] = (1-epsilon)*Sigma[:,:,0] + epsilon*Sigma[:,:,n]
    Sigma = Sigma.to(device='cuda')
    return Sigma



""" def make_S(Sigma, T):
    _, K, N = Sigma.size()
    S = torch.zeros(N, T, K)
    mean = torch.zeros(K)
    for n in range(N):
        S[n,:,:] = torch.tensor(np.random.multivariate_normal(mean,Sigma[:,:,n],T))
        #S[n, :, :] = torch.normal(mean, torch.sqrt(Sigma[:, :, n]), (T, K))
    return S """

def make_S(Sigma, T):
    _, K, N = Sigma.size()
    S = torch.zeros(N, T, K, device='cuda')
    mean = torch.zeros(K, device='cuda')
    
    for n in range(N):
        cov_matrix = Sigma[:, :, n]
        mvn = torch.distributions.MultivariateNormal(mean, cov_matrix)
        S[n, :, :] = mvn.sample((T,))
    return S


def make_X(S,A):
    X = torch.einsum('MNK,NTK -> MTK',A,S)
    return X


def generate_whitened_problem(T,K,N,epsilon=1,rho_bounds=[0.4,0.6],lambda_=0.25): #, idx_W=None):
    A = make_A(K,N)
    # A = full_to_blocks(A,idx_W,K)
    Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,U = whiten_data(X)
    A_ = torch.einsum('nNk,Nvk->nvk', U, A)
    X_ = X_.to(device='cuda')
    A_ = A_.to(device='cuda')
    return X_,A_


def get_metaparameters(rhos,lambdas):
    metaparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            metaparameters_multiparam.append((rho_bounds,lambda_))
    return metaparameters_multiparam


