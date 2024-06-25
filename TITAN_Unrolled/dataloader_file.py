from data import *
from functions import *
from torch.utils.data import DataLoader

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
size = 1



data = MyDataset(T,K,N,metaparameters_multiparam,size)
dataloader = DataLoader(data, batch_size=1, shuffle=True)


