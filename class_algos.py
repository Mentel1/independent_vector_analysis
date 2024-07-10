import os
import numpy as np
from time import time
from algorithms.iva_g_numpy import *
from algorithms.iva_g_torch import *
from algorithms.algebra_toolbox_torch import *
from algorithms.algebra_toolbox_numpy import *
from algorithms.titan_iva_g_reg_numpy import *
from algorithms.titan_iva_g_reg_torch import *
# from TITAN_Unrolled. 


class IvaGAlgorithms:

    def __init__(self,name,legend,color,library):
        self.name = name
        self.legend = legend
        self.color = color
        self.library = library
        self.results = None
        self.times = None

    def set_up_for_benchmark_experiment(self,parameters_dimensions):
        self.results = np.zeros(parameters_dimensions)
        self.times = np.zeros(parameters_dimensions)

    def solve_with_jisi(self,X,A):
        pass
    
    def solve_with_cost(self,X):
        pass

    def solve(self,X,Winit,Cinit):
        pass

    def fill_experiment(self,X,A,coordinates,Winit=None,Cinit=None):
        self.times[coordinates] -= time()
        W = self.solve(X,Winit,Cinit)
        self.times[coordinates] += time()
        if self.library == 'torch':
            A = torch.tensor(A)
            self.results[coordinates] = joint_isi_torch(W,A)
        elif self.library == 'numpy':
            self.results[coordinates] = joint_isi_numpy(W,A)

    def fill_from_folder(self,foldername,meta_parameters,meta_parameters_titles,common_parameters,N_exp):
        Ks,Ns = common_parameters
        dimensions = len(meta_parameters),len(Ks),len(Ns),N_exp
        self.set_up_for_benchmark_experiment(dimensions)
        for a,metaparam in enumerate(meta_parameters):
            metaparam_title = meta_parameters_titles[a]   
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    filepath = foldername + '/{}/N = {} K = {}'.format(metaparam_title,N,K)
                    if os.path.exists(filepath):
                        self.results[a,ik,jn,:] = np.fromfile(filepath+'/results_'+self.name,sep=',')
                        self.times[a,ik,jn,:] = np.fromfile(filepath+'/times_'+self.name,sep=',')

class IvaG(IvaGAlgorithms):

    def __init__(self,color='b',name='IVA-G-N',legend='IVA-G-N',opt_approach='newton',max_iter=20000,crit_ext=1e-6,library='numpy'):
        super().__init__(name=name,legend=legend,color=color,library=library)
        self.alternated = False
        self.opt_approach = opt_approach
        self.max_iter = max_iter
        self.crit_ext = crit_ext
        self.library = library

    def solve(self,X,Winit,Cinit):
            _,_,K = X.shape
            for k in range(K):
                Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
            if self.library == 'numpy':
                W,_,_,_,_ = iva_g_numpy(X,opt_approach=self.opt_approach,W_diff_stop=self.crit_ext,
                                max_iter=self.max_iter,W_init=Winit)
            elif self.library == 'torch':
                Winit = torch.tensor(Winit)
                X = torch.from_numpy(X)
                W,_,_,_,_ = iva_g_torch(X,opt_approach=self.opt_approach,W_diff_stop=self.crit_ext,
                                max_iter=self.max_iter,W_init=Winit)
            return W  

    def solve_with_jisi(self,X,A):
        if self.library == 'numpy':
            _,_,_,jisi,_  = iva_g_numpy(X,opt_approach=self.opt_approach,jdiag_initW=False,W_diff_stop=self.crit_ext,
                            max_iter=self.max_iter,A=A)
        elif self.library == 'torch':
            A = torch.tensor(A)
            X = torch.tensor(X)
            _,_,_,jisi,_  = iva_g_torch(X,opt_approach=self.opt_approach,jdiag_initW=False,W_diff_stop=self.crit_ext,
                            max_iter=self.max_iter,A=A)

        return jisi
    
    def solve_with_cost(self,X):
        if self.library == 'numpy':
            _,cost,_,_,_ = iva_g_numpy(X,opt_approach=self.opt_approach,jdiag_initW=False,W_diff_stop=self.crit_ext,
                            max_iter=self.max_iter)
        elif self.library == 'torch':
            _,cost,_,_,_ = iva_g_torch(X,opt_approach=self.opt_approach,jdiag_initW=False,W_diff_stop=self.crit_ext,
                            max_iter=self.max_iter)
            X = torch.tensor(X)
        return cost

'''   
class PalmIvaG(IvaGAlgorithms):

    def __init__(self,color,name='palm',alpha=1,max_iter=20000,max_iter_int=50,crit_ext=1e-9,crit_int=1e-9,
                 gamma_w=0.99,gamma_c=1.99,seed=None):
        super().__init__(name=name,color=color,max_iter=max_iter,crit_ext=crit_ext)
        self.crit_int = crit_int
        self.max_iter_int = max_iter_int
        self.alpha = alpha
        self.alternated = True
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.seed = seed

    def solve(self,X,Winit,Cinit):
        W,_,_ = palm_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                               crit_ext=self.crit_ext,crit_int=self.crit_int,
                               max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                               Winit=Winit,Cinit=Cinit,seed=self.seed)
        return W  
    
    def solve_with_isi(self,X,A,Winit,Cinit):
        _,_,_,isi = palm_iva_g_reg(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                               crit_ext=self.crit_ext,crit_int=self.crit_int,
                               max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                               Winit=Winit,Cinit=Cinit,seed=self.seed,track_isi=True,B=A)
    
        return isi
'''

class TitanIvaG(IvaGAlgorithms):    

    def __init__(self,color,name='titan',legend='TITAN-IVA-G',nu=0.5,max_iter=20000,max_iter_int=15,max_iter_int_C=1,
                 crit_ext=1e-10,crit_int=1e-10,gamma_w=0.99,gamma_c=1,alpha=1,seed=None,library='numpy'):
        super().__init__(name=name,legend=legend,color=color,library=library)
        self.crit_int = crit_int
        self.crit_ext = crit_ext
        self.max_iter = max_iter
        self.max_iter_int = max_iter_int
        self.max_iter_int_C = max_iter_int_C
        self.nu = nu
        self.alpha = alpha
        self.alternated = True
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.seed = seed
        self.library = library

    def solve(self,X,Winit=None,Cinit=None):
        if self.library == 'numpy':
            W,_,_,_ = titan_iva_g_reg_numpy(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     max_iter_int_C=self.max_iter_int_C,nu=self.nu,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed)
        elif self.library == 'torch':
            W,_,_,_ = titan_iva_g_reg_torch(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     max_iter_int_C=self.max_iter_int_C,nu=self.nu,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed)
        return W 

    def solve_with_jisi(self,X,A,Winit=None,Cinit=None):
        if self.library == 'numpy':
            _,_,_,_,jisi = titan_iva_g_reg_numpy(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_jisi=True,B=A)
        elif self.library == 'troch':
             _,_,_,_,jisi = titan_iva_g_reg_torch(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_jisi=True,B=A)
        

        return jisi

    def solve_with_cost(self,X,Winit=None,Cinit=None):
        if self.library == 'numpy':
            _,_,_,times,cost = titan_iva_g_reg_numpy(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_cost=True)
        elif self.library == 'torch':
            _,_,_,times,cost = titan_iva_g_reg_torch(X,alpha=self.alpha,gamma_w=self.gamma_w,gamma_c=self.gamma_c,
                                     crit_ext=self.crit_ext,crit_int=self.crit_int,nu=self.nu,
                                     max_iter=self.max_iter,max_iter_int=self.max_iter_int,
                                     Winit=Winit,Cinit=Cinit,seed=self.seed,track_cost=True)
        
        return times,cost   
    

# N = 10
# K = 10
# T = 10000
# rho_bounds = [0.2,0.3]
# lambda_ = 0.25
# epsilon = 1
# X,A = generate_whitened_problem(T,K,N,epsilon,rho_bounds,lambda_)
# Winit = make_A(K,N)
# Cinit = make_Sigma(K,N,rank=K+10)

# output_folder = 'empirical convergence'
# os.makedirs(output_folder,exist_ok=True)

# _,_,_,times_titan,cost_titan,jisi_titan = titan_iva_g_reg(X.copy(),track_cost=True,track_jisi=True,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)
# for k in range(K):
#             Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
# _,_,_,jisi_n,times_n = iva_g(X.copy(),opt_approach='newton',A=A,W_init=Winit.copy(),W_diff_stop=1e-7)
# _,_,_,jisi_v,times_v = iva_g(X.copy(),opt_approach='gradient',A=A,W_init=Winit.copy())

# np.array(times_titan).tofile(output_folder+'/times_titan',sep=',')
# np.array(cost_titan).tofile(output_folder+'/cost_titan',sep=',')
# np.array(jisi_titan).tofile(output_folder+'/jisi_titan',sep=',')
# np.array(times_v).tofile(output_folder+'/times_v',sep=',')
# np.array(jisi_v).tofile(output_folder+'/jisi_v',sep=',')
# np.array(times_n).tofile(output_folder+'/times_n',sep=',')
# np.array(jisi_n).tofile(output_folder+'/jisi_n',sep=',')