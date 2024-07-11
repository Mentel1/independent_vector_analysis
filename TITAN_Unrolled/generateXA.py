import torch
from model import myModel
from data import *







# Hyperparameters

T = 10000
K = 2
N = 3


lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']


X,A = generate_whitened_problem(T,K,N,epsilon=1,rho_bounds=rho_bounds_2,lambda_=lambda_1)
Winit = make_A(K,N)
Cinit = make_Sigma(K,N,rank=K+10)

# Save X and A to a file
print(X)
torch.save(X, 'Unrolled-TITAN/TITAN_Unrolled/X.pt')
torch.save(A, 'Unrolled-TITAN/TITAN_Unrolled/A.pt')
print(X.shape)

torch.save(Winit, 'Unrolled-TITAN/TITAN_Unrolled/Winit.pt')
torch.save(Cinit, 'Unrolled-TITAN/TITAN_Unrolled/Cinit.pt')   

