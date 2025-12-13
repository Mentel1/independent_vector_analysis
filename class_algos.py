import os
import sys
import numpy as np
from time import time
from algorithms.iva_g_numpy import *
from algorithms.iva_g_torch import *
from algorithms.algebra_toolbox_torch import *
from algorithms.algebra_toolbox_numpy import *
from algorithms.titan_iva_g_reg_numpy import *
from algorithms.titan_iva_g_reg_torch import *
from algorithms.titan_iva_g_reg_numpy_exact_C import *
from algorithms.AuxIVA_ISS_fakufaku.piva.auxiva_iss import auxiva_iss_py

class IvaGAlgorithms:

    def __init__(self,name,legend,color,library,down_sample=False,num_samples=None,inflate=False,lambda_inflate=1e-3):
        self.name = name
        self.legend = legend
        self.color = color
        self.library = library
        self.down_sample = down_sample
        self.num_samples = num_samples
        self.inflate = inflate
        self.lambda_inflate = lambda_inflate
        self.results = {}

    def solve(self,X):
        pass
    
    def to_dict(self):
        """Convertit l'algo en dictionnaire sérialisable"""
        return {}
    
    @classmethod
    def from_dict(cls, config):
        config['color'] = tuple(config['color'])
        config = config.copy()
        class_name = config.pop('class')  # Ex: "TitanIvaG"
        current_module = sys.modules[__name__]
        actual_cls = getattr(current_module, class_name)
        return actual_cls(**config)

    def fill_experiment(self,X,A,exp,Winit=None,Cinit=None):
        res = self.solve(X,Winit,Cinit)
        self.results['total_times'][exp] = res['times'][-1]
        W = res['W']
        if self.library == 'torch':
            A = torch.tensor(A)
            self.results['final_jisi'][exp] = joint_isi_torch(W,A)
        elif self.library == 'numpy':
            self.results['final_jisi'][exp] = joint_isi_numpy(W,A)

    def fill_from_folder(self,output_path_individual):
        for result in ['total_times','jisi_final']:
            res_path = os.path.join(output_path_individual,self.name,result)
            self.results[result] = np.fromfile(res_path,sep=',')

class IvaG(IvaGAlgorithms):

    def __init__(self,color='b',name='IVA-G-N',legend='IVA-G-N',opt_approach='newton',max_iter=5000,crit_ext=1e-6,library='numpy',fast=False,down_sample=False,num_samples=10000,Jdiag_init=False,inflate=False,lambda_inflate=1e-3):
        super().__init__(name=name,legend=legend,color=color,library=library,down_sample=down_sample,num_samples=num_samples,inflate=inflate,lambda_inflate=lambda_inflate)
        self.opt_approach = opt_approach
        self.crit_ext = crit_ext
        self.fast = fast
        self.Jdiag_init = Jdiag_init
        self.max_iter = max_iter

    def to_dict(self):
        return {'class': self.__class__.__name__,'color': self.color,'name': self.name,'legend': self.legend,'library': self.library,'down_sample': self.down_sample,'num_samples': self.num_samples,'inflate': self.inflate,'lambda_inflate': self.lambda_inflate,'opt_approach': self.opt_approach,'crit_ext': self.crit_ext,'fast': self.fast,'Jdiag_init': self.Jdiag_init,'max_iter': self.max_iter}
        
    def _get_base_params(self):
        return {'W_diff_stop': self.crit_ext,'max_iter': self.max_iter,'down_sample':self.down_sample,'num_samples':self.num_samples,'inflate': self.inflate,'lambda_inflate': self.lambda_inflate,'opt_approach':self.opt_approach,'Jdiag_init': self.Jdiag_init,}
    
    def solve(self,X,Winit=None,Cinit=None,track_jisi=False,B=None,track_costs=False,track_diffs=False,track_schemes=False,track_shifts=False,track_times=True):
            self.normalize_Winit(X, Winit)
            params = self._get_base_params()
            params.update({'Winit':Winit,'A':B})
            
            if self.library == 'numpy':
                if self.fast:
                    return fast_iva_g_numpy(X,**params)
                else:
                    return iva_g_numpy(X,**params)
            elif self.library == 'torch':
                Winit = torch.tensor(Winit)
                X = torch.from_numpy(X)
                if self.fast:
                    return fast_iva_g_torch(X,**params)
                else:
                    return iva_g_torch(X,**params)

    def normalize_Winit(X, Winit):
        _,_,K = X.shape
        for k in range(K):
            Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k]) 
    

class TitanIvaG(IvaGAlgorithms):    

    def __init__(self,color,name='titan',legend='TITAN-IVA-G',library='numpy',inflate=False,lambda_inflate=1e-3,down_sample=False,num_samples=10000,nu=0.5,max_iter=20000,max_iter_int_W=15,max_iter_int_C=1,crit=1e-10,epsilon=1e-12,zeta=1e-3,gamma_w=0.99,gamma_c=1,alpha=1,init_method='random',seed=None,boost=False,exactC=False):
        super().__init__(name=name,legend=legend,color=color,library=library,down_sample=down_sample,num_samples=num_samples,inflate=inflate,lambda_inflate=lambda_inflate)
        self.crit_int = crit # remettre les deux arguments séparés après avoir fini le manuscrit
        self.crit_ext = crit
        self.max_iter = max_iter
        self.max_iter_int_W = max_iter_int_W
        self.max_iter_int_C = max_iter_int_C
        self.nu = nu
        self.alpha = alpha
        self.epsilon = epsilon
        self.zeta = zeta
        self.gamma_w = gamma_w
        self.gamma_c = gamma_c
        self.init_method = init_method
        self.seed = seed
        self.boost = boost
        self.exactC = exactC

    def to_dict(self):
        return {'class': self.__class__.__name__, 'color': self.color,'name': self.name,'legend': self.legend,'library': self.library,'down_sample': self.down_sample,'num_samples': self.num_samples,'inflate': self.inflate,'lambda_inflate': self.lambda_inflate,'crit_int': self.crit_int,'crit_ext': self.crit_ext,'max_iter_int_W': self.max_iter_int_W,'max_iter_int_C': self.max_iter_int_C,'max_iter': self.max_iter,'nu': self.nu,'alpha': self.alpha,'epsilon': self.epsilon,'zeta': self.zeta, 'gamma_w': self.gamma_w,'gamma_c': self.gamma_c,'init_method': self.init_method,'seed': self.seed,'boost': self.boost,'exactC': self.exactC}
    
    @classmethod
    def from_dict(cls, config):
        """Reconstruit un algo depuis un dictionnaire"""
        return cls(**config)
    
    def _get_base_params(self):
        return {'alpha': self.alpha,'gamma_w': self.gamma_w,'gamma_c': self.gamma_c,'crit_ext': self.crit_ext,'crit_int': self.crit_int,'epsilon': self.epsilon,'zeta':self.zeta,'nu': self.nu,'max_iter': self.max_iter,'max_iter_int_W': self.max_iter_int_W,'max_iter_int_C': self.max_iter_int_C,'seed': self.seed,'boost':self.boost,'inflate':self.inflate,'lambda_inflate':self.lambda_inflate,'down_sample':self.down_sample,'num_samples':self.num_samples}
              
    def solve(self,X,Winit=None,Cinit=None,track_jisi=False,B=None,track_costs=False,track_diffs=False,track_schemes=False,track_shifts=False,track_times=True):
        params = self._get_base_params()
        params.update({'Winit':Winit,'Cinit':Cinit,'track_jisi':track_jisi,'B':B,'track_costs':track_costs,'track_diffs':track_diffs,'track_schemes':track_schemes,'track_shifts':track_shifts,'track_times':track_times})    
        if self.exactC:
            return titan_iva_g_reg_numpy_exactC(X,**params)
        else:
            if self.library == 'numpy':
                return titan_iva_g_reg_numpy(X,**params)
            elif self.library == 'torch':
                return titan_iva_g_reg_torch(X,**params)

 
class AuxIVA_ISS(IvaGAlgorithms):

    def __init__(self,color,name='aux_iva_iss',legend='AuxIVA-ISS',n_iter=200,library='numpy',
                 backend='py',model='laplace',proj_back=False):
        super().__init__(name=name,legend=legend,color=color,library=library)
        self.n_iter = n_iter
        self.backend = backend
        self.model = model
        self.proj_back = proj_back
        

    def solve(self,X,Winit=None,Cinit=None):
        X = np.moveaxis(X, 0, 2)
        _,W = auxiva_iss_py(X,n_iter=self.n_iter, proj_back=self.proj_back, model=self.model, return_filters=True)
        W = np.moveaxis(W,0,2)
        return W