import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cProfile
from class_exp import *
from class_algos import *
from algorithms.iva_g_numpy import *
from algorithms.titan_iva_g_reg_numpy import *
from algorithms.titan_iva_g_reg_torch import *
import torch

label_size = 60
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

# Function to generate metaparameters for the multiparameter experiment
def get_metaparameters(rhos,lambdas):
    metaparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            metaparameters_multiparam.append((rho_bounds,lambda_))
    return metaparameters_multiparam


def create_algos_titanIVAG(varying_param, values, color_bounds=[(0.2,1,0.2),(0.2,0.2,1)],base_params={},basename=''):
    algos = []
    nval = len(values)
    for i, value in enumerate(values):
        params = base_params.copy()
        params[varying_param] = value
        t = i / (nval - 1)
        params['color'] = tuple((1 - t) * c0 + t * c1 for c0, c1 in zip(color_bounds[0], color_bounds[1]))
        params['name'] = basename + '_' + varying_param + '=' + str(value)      
        algos.append(TitanIvaG(**params))
    return algos


#================================================================================================
# MAIN EXPERIMENT (MULTIPARAMETER)
#================================================================================================

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_2,lambda_1]
metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']
metaparameters_base = get_metaparameters([[0.4,0.6]],[0.1])
metaparameters_base_titles = ['Base_case']
# metaparameters_identifiability = [1e-2,1e-1,1]
# metaparameters_titles_identifiability = ['low identifiability','medium identifiability','high identifiability']

Ks = [5,10,20]
Ns = [10,20,30] 
common_parameters = [Ks,Ns]

algos = create_algos_titanIVAG('gamma_w',[0.5,0.9,0.99],base_params={'nu':0,'max_iter_int_W':1})
exp = ComparisonExperimentIvaG('CompareGammaW',metaparameters_base,metaparameters_base_titles,common_parameters,algos,N_exp=10)
exp.compute_empirical_convergence()



# algos = create_algos_titanIVAG('alpha',[0.01,0.05,0.1,0.5,1,5,10],base_params={'nu':0,'max_iter_int_W':1})
# exp = ComparisonExperimentIvaG('CompareAlpha',metaparameters_base,metaparameters_base_title,common_parameters,algos,N_exp=20)
# exp.compute()




# algo_titan = TitanIvaG((0,0.4,0),name='titan',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
# algo_palm = TitanIvaG((0,0.4,0),name='palm',legend='PALM-IVA-G',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy')
# algo_palm_torch = TitanIvaG((0,0.8,0),name='palm_torch',legend='PALM-IVA-G-TORCH',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='torch')
# algo_iva_g_n = IvaG((0.5,0,0),name='iva_g_n',legend='IVA-G-N',crit_ext=1e-7,opt_approach='newton',library='numpy')
# algo_iva_g_v = IvaG((0.5,1,0),name='iva_g_v',legend='IVA-G-V',crit_ext=1e-6,opt_approach='gradient',library='numpy')
# algo_palm_boost = TitanIvaG((0,0.4,0),name='palm_boost',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy',boost=True)


# algos = [algo_palm,algo_iva_g_v,algo_iva_g_n]


# exp1 = ComparisonExperimentIvaG('rank effect',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters,'effective rank',title_fontsize=50,legend_fontsize=6,N_exp=3,charts=False,legend=False)
# exp2 = ComparisonExperimentIvaG('palm part 2',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters_2,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=20,charts=False,legend=False)
# exp1.compute()
# exp2.get_data_from_folder('2024-05-16_02-22')
# exp1.make_table()

# exp2.make_charts(full=True)


#================================================================================================
# EXPERIMENT FOR THE STEPSIZES
#================================================================================================


# algo_titan_50_1 = TitanIvaG((0,0.4,0.4),name='titan_50_1',alpha=1,gamma_w=0.5,gamma_c=1,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.5, \gamma_C = 1$')
# algo_titan_50_19 = TitanIvaG((0,0.4,0.7),name='titan_50_19',alpha=1,gamma_w=0.5,gamma_c=1.9,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.5, \gamma_C = 1.9$')
# algo_titan_50_199 = TitanIvaG((0,0.4,1),name='titan_50_199',alpha=1,gamma_w=0.5,gamma_c=1.99,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.5, \gamma_C = 1.99$')
# algo_titan_90_1 = TitanIvaG((0,0.7,0.4),name='titan_90_1',alpha=1,gamma_w=0.9,gamma_c=1,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.9, \gamma_C = 1$')
# algo_titan_90_19 = TitanIvaG((0,0.7,0.7),name='titan_90_19',alpha=1,gamma_w=0.9,gamma_c=1.9,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.9, \gamma_C = 1.9$')
# algo_titan_90_199 = TitanIvaG((0,0.7,1),name='titan_90_199',alpha=1,gamma_w=0.9,gamma_c=1.99,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.9, \gamma_C = 1.99$')
# algo_titan_99_1 = TitanIvaG((0,1,0.4),name='titan_99_1',alpha=1,gamma_w=0.99,gamma_c=1,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.99, \gamma_C = 1$')
# algo_titan_99_19 = TitanIvaG((0,1,0.7),name='titan_99_19',alpha=1,gamma_w=0.99,gamma_c=1.9,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.99, \gamma_C = 1.9$')
# algo_titan_99_199 = TitanIvaG((0,1,1),name='titan_99_199',alpha=1,gamma_w=0.99,gamma_c=1.99,crit_ext=0,crit_int=0,library='numpy',max_iter=2000,max_iter_int=1,max_iter_int_C=1,legend='$\gamma_W = 0.99, \gamma_C = 1.99$')

# algos = [algo_titan_50_1,algo_titan_50_19,algo_titan_50_199,algo_titan_90_1,algo_titan_90_19,algo_titan_90_199,algo_titan_99_1,algo_titan_99_19,algo_titan_99_199]
# exp2 = ComparisonExperimentIvaG('stepsize experiment',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters_1,'multiparam',title_fontsize=20,legend_fontsize=6,T=10000,N_exp=10,charts=False,legend=False)
# exp2.compute_empirical_convergence()


#================================================================================================
# EMPIRICAL CONVERGENCE FOR THE STEPSIZES
#================================================================================================


# N = 20
# K = 20
# T = 10000
# rho_bounds = [0.4,0.6]
# lambda_ = 0.1
# epsilon = 1
# X,A = generate_whitened_problem(T,K,N,epsilon,rho_bounds,lambda_)
# Winit = make_A(K,N)
# Cinit = make_Sigma(K,N,rank=K+10)

# output_folder = 'Result_data/empirical convergence'
# os.makedirs(output_folder,exist_ok=True)

# _,_,_,times_palm_50,cost_palm_50,jisi_palm_50 = titan_iva_g_reg_numpy(X.copy(),track_cost=True,track_jisi=True,gamma_w=0.5,gamma_c=1.99,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)
# _,_,_,times_palm_90,cost_palm_90,jisi_palm_90 = titan_iva_g_reg_numpy(X.copy(),track_cost=True,track_jisi=True,gamma_w=0.5,gamma_c=1.99,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)
# _,_,_,times_palm_99,cost_palm_99,jisi_palm_99 = titan_iva_g_reg_numpy(X.copy(),track_cost=True,track_jisi=True,gamma_w=0.99,gamma_c=1.99,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)


# np.array(times_palm).tofile(output_folder+'/times_palm',sep=',')
# np.array(cost_palm).tofile(output_folder+'/cost_palm',sep=',')
# np.array(jisi_palm).tofile(output_folder+'/jisi_palm',sep=',')


#================================================================================================
# EXPERIMENT FOR AuxIVA_ISS
#================================================================================================
# algo_aux_iva_iss_l = AuxIVA_ISS((0.4,0,0.8),name='aux_iva_iss_l',legend='AuxIVA-ISS-L',library='numpy',proj_back=False,model='laplace',n_iter=10000)
# # algo_aux_iva_iss_g = AuxIVA_ISS((0.8,0,0.8),name='aux_iva_iss_g',legend='AuxIVA-ISS-G',library='numpy',proj_back=False,model='gauss')
# algos = [algo_aux_iva_iss_l] #, algo_aux_iva_iss_g]  

# exp4 = ComparisonExperimentIvaG('AuxIVA-ISS',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,T=10000,N_exp=10,charts=False,legend=False)
# exp4.compute()

#================================================================================================
# EXPERIMENT FOR THE EFFECT OF EXACT C UPDATES
#================================================================================================

# algo_palm = TitanIvaG((0,0.4,0),name='palm',legend='PALM-IVA-G',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy')
# algo_palm_exactC = TitanIvaG((0,0.4,0),name='palm_exactC',legend='PALM-IVA-G-ExactC',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy',exactC=True)
# algos = [algo_palm,algo_palm_exactC]

# exp3 = ComparisonExperimentIvaG('experiment for exact C',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=40,charts=False,legend=False)
# exp3.compute()

#================================================================================================
# EXPERIMENT FOR THE EFFECT OF RANK (to be completed)
#================================================================================================


# effective_ranks = [None]
# metaparameters_multiparam = effective_ranks


#================================================================================================
# EXPERIMENT FOR THE EFFECT OF EPSILON
#================================================================================================

# lambda_ = 0.25
# rho_bounds = [0.2,0.3]
# lambdas = [lambda_]
# rhos = [rho_bounds]
# Ns = [10]
# Ks = [10]
# T = 10000
# epsilon_exponents = [-12,-9,-6,-3,-2,-1]
# metaparameters_multiparam = get_metaparameters(rhos,lambdas)
# common_parameters_1 = [Ks,Ns]
# metaparameters_titles_multiparam = ['Case A']

# algo_titan_12 = TitanIvaG((0,0.2,0),name='eps = 1e-12',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-12)
# algo_titan_9 = TitanIvaG((0,0.4,0),name='eps = 1e-9',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-9)
# algo_titan_6 = TitanIvaG((0,0.6,0),name='eps = 1e-6',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-6)
# algo_titan_3 = TitanIvaG((0,0.8,0),name='eps = 1e-3',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-3)
# algo_titan_2 = TitanIvaG((0,0.9,0),name='eps = 1e-2',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-2)
# algo_titan_1 = TitanIvaG((0,1,0),name='eps = 1e-1',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',epsilon=1e-1)

# algos = [algo_titan_12,algo_titan_9,algo_titan_6,algo_titan_3,algo_titan_2,algo_titan_1]

# exp1 = ComparisonExperimentIvaG('effect of epsilon',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
#                                 common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=10,charts=False,legend=False)
                               
# exp1.compute()
# exp1.make_table()

#================================================================================================
# ROBUSTNESS TO INFLATION OF Rx
#================================================================================================

# inflate_lambdas = [0,1e-3,1e-2,1e-1,1,10]

# algo_palm_inflate_0 = TitanIvaG((0,0.2,0),name='palm_inflate_0',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=False)
# algo_palm_inflate_0001 = TitanIvaG((0,0.4,0),name='palm_inflate_0001',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=True,lambda_inflate=1e-3)
# algo_palm_inflate_001 = TitanIvaG((0,0.6,0),name='palm_inflate_001',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=True,lambda_inflate=1e-2)
# algo_palm_inflate_01 = TitanIvaG((0,0.7,0),name='palm_inflate_01',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=True,lambda_inflate=1e-1)
# algo_palm_inflate_1 = TitanIvaG((0,0.8,0),name='palm_inflate_1',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=True,lambda_inflate=1)
# algo_palm_inflate_10 = TitanIvaG((0,1,0),name='palm_inflate_10',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',inflate=True,lambda_inflate=10)

# algos = [algo_palm_inflate_0001,algo_palm_inflate_001,algo_palm_inflate_01,algo_palm_inflate_1,algo_palm_inflate_10]

# exp1 = ComparisonExperimentIvaG('robustness to errors on Rx',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
#                                 common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=100,charts=False,legend=False)
# exp1.compute()
# exp1.make_table()

#================================================================================================
# ROBUSTNESS TO DOWNSAMPLING
#================================================================================================

# algo_palm_10000 = TitanIvaG((0,0.2,0),name='palm_10000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=False)
# algo_palm_5000 = TitanIvaG((0,0.4,0),name='palm_5000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=5000)
# algo_palm_1000 = TitanIvaG((0,0.6,0),name='palm_1000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=1000)
# algo_palm_500 = TitanIvaG((0,0.7,0),name='palm_500',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=500)
# algo_palm_200 = TitanIvaG((0,0.8,0),name='palm_200',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=200)
# algo_palm_150 = TitanIvaG((0,0.8,0),name='palm_150',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=150)
# algo_palm_120 = TitanIvaG((0,0.8,0),name='palm_120',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=120)
# algo_palm_100 = TitanIvaG((0,0.9,0),name='palm_100',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=100)
# algo_palm_50 = TitanIvaG((0,1,0),name='palm_50',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,max_iter=5000,library='numpy',down_sample=True,num_samples=50)

# algo_iva_g_n_10000 = IvaG((0,0.2,0),name='iva_g_n_10000',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=False)
# algo_iva_g_n_5000 = IvaG((0,0.4,0),name='iva_g_n_5000',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=5000)
# algo_iva_g_n_1000 = IvaG((0,0.6,0),name='iva_g_n_1000',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=1000)
# algo_iva_g_n_500 = IvaG((0,0.7,0),name='iva_g_n_500',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=500)
# algo_iva_g_n_200 = IvaG((0,0.8,0),name='iva_g_n_200',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=200)
# algo_iva_g_n_150 = IvaG((0,0.8,0),name='iva_g_n_150',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=150)
# algo_iva_g_n_120 = IvaG((0,0.8,0),name='iva_g_n_120',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=120)
# algo_iva_g_n_100 = IvaG((0,0.9,0),name='iva_g_n_100',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=100)
# algo_iva_g_n_50 = IvaG((0,1,0),name='iva_g_n_50',crit_ext=1e-7,opt_approach='newton',library='numpy',down_sample=True,num_samples=50)

# algo_iva_g_v_10000 = IvaG((0,0.2,0),name='iva_g_v_10000',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=False)
# algo_iva_g_v_5000 = IvaG((0,0.4,0),name='iva_g_v_5000',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=5000)
# algo_iva_g_v_1000 = IvaG((0,0.6,0),name='iva_g_v_1000',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=1000)
# algo_iva_g_v_500 = IvaG((0,0.7,0),name='iva_g_v_500',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=500)
# algo_iva_g_v_200 = IvaG((0,0.8,0),name='iva_g_v_200',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=200)
# algo_iva_g_v_150 = IvaG((0,0.8,0),name='iva_g_v_150',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=150)
# algo_iva_g_v_120 = IvaG((0,0.8,0),name='iva_g_v_120',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=120)
# algo_iva_g_v_100 = IvaG((0,0.9,0),name='iva_g_v_100',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=100)
# algo_iva_g_v_50 = IvaG((0,1,0),name='iva_g_v_50',crit_ext=1e-7,opt_approach='gradient',library='numpy',down_sample=True,num_samples=50)

# algos = [algo_iva_g_n_10000,algo_iva_g_v_10000,algo_iva_g_n_5000,algo_iva_g_v_5000,algo_iva_g_n_1000,algo_iva_g_v_1000,algo_iva_g_n_500,algo_iva_g_v_500,algo_iva_g_n_200,algo_iva_g_v_200,algo_iva_g_n_150,algo_iva_g_v_150,algo_iva_g_n_120,algo_iva_g_v_120,algo_iva_g_n_100,algo_iva_g_v_100,algo_iva_g_n_50,algo_iva_g_v_50]

# exp1 = ComparisonExperimentIvaG('robustness to downsampling iva_g',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
#                                 common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=100,charts=False,legend=False)
# exp1.compute()
# exp1.make_table()


#================================================================================================
# ROBUSTNESS TO DOWNSAMPLING + INFLATION OF Rx
#================================================================================================

# algo_palm_10000 = TitanIvaG((0,0.2,0),name='palm_10000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=False)
# algo_palm_5000 = TitanIvaG((0,0.4,0),name='palm_5000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=5000)
# algo_palm_1000 = TitanIvaG((0,0.6,0),name='palm_1000',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=1000)
# algo_palm_500 = TitanIvaG((0,0.7,0),name='palm_500',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=500)
# algo_palm_200 = TitanIvaG((0,0.8,0),name='palm_200',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=200)
# algo_palm_150 = TitanIvaG((0,0.8,0),name='palm_150',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=150)
# algo_palm_120 = TitanIvaG((0,0.8,0),name='palm_120',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=120)
# algo_palm_100 = TitanIvaG((0,0.9,0),name='palm_100',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=100)
# algo_palm_50 = TitanIvaG((0,1,0),name='palm_50',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=50)
# algo_palm_10000_inflate = TitanIvaG((0,0.2,0),name='palm_10000_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=False,inflate=True,lambda_inflate=1)
# algo_palm_5000_inflate = TitanIvaG((0,0.4,0),name='palm_5000_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=5000,inflate=True,lambda_inflate=1)
# algo_palm_1000_inflate = TitanIvaG((0,0.6,0),name='palm_1000_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=1000,inflate=True,lambda_inflate=1)
# algo_palm_500_inflate = TitanIvaG((0,0.7,0),name='palm_500_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=500,inflate=True,lambda_inflate=1)
# algo_palm_200_inflate = TitanIvaG((0,0.8,0),name='palm_200_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=200,inflate=True,lambda_inflate=1)
# algo_palm_150_inflate = TitanIvaG((0,0.8,0),name='palm_150_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=150,inflate=True,lambda_inflate=1)
# algo_palm_120_inflate = TitanIvaG((0,0.8,0),name='palm_120_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=120,inflate=True,lambda_inflate=1)
# algo_palm_100_inflate = TitanIvaG((0,0.9,0),name='palm_100_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=100,inflate=True,lambda_inflate=1)
# algo_palm_50_inflate = TitanIvaG((0,1,0),name='palm_50_inflate',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy',down_sample=True,num_samples=50,inflate=True,lambda_inflate=1)

# algos = [algo_palm_10000,algo_palm_10000_inflate,algo_palm_5000,algo_palm_5000_inflate,algo_palm_1000,algo_palm_1000_inflate,algo_palm_500,algo_palm_500_inflate,algo_palm_200,algo_palm_200_inflate,algo_palm_150,algo_palm_150_inflate,algo_palm_120,algo_palm_120_inflate,algo_palm_100,algo_palm_100_inflate,algo_palm_50,algo_palm_50_inflate]

# exp1 = ComparisonExperimentIvaG('robustness to downsampling + inflation',algos,metaparameters_multiparam,metaparameters_titles_multiparam,common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=10,charts=False,legend=False)
# exp1.compute()
# exp1.make_table()

#================================================================================================
# ANALYSIS OF THE SLOWEST SUBPROCESS
#================================================================================================

# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp1.compute()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()


#================================================================================================
# EXPERIMENT FOR EMPIRICAL CONVERGENCE
#================================================================================================

# N = 20
# K = 20
# T = 10000
# rho_bounds = [0.2,0.3]
# lambda_ = 0.25
# epsilon = 1
# X,A = generate_whitened_problem(T,K,N,epsilon,rho_bounds,lambda_)
# Winit = make_A(K,N)
# Cinit = make_Sigma(K,N,rank=K+10)

# output_folder = 'Result_data/empirical convergence'
# os.makedirs(output_folder,exist_ok=True)

# _,_,_,times_palm,cost_palm,jisi_palm = titan_iva_g_reg_numpy(X.copy(),track_cost=True,track_jisi=True,gamma_c=1.99,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)
# for k in range(K):
#     Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
# _,_,_,jisi_n,times_n = iva_g_numpy(X.copy(),opt_approach='newton',A=A,W_init=Winit.copy(),W_diff_stop=1e-7)
# _,_,_,jisi_v,times_v = iva_g_numpy(X.copy(),opt_approach='gradient',A=A,W_init=Winit.copy())

# np.array(times_palm).tofile(output_folder+'/times_palm',sep=',')
# np.array(cost_palm).tofile(output_folder+'/cost_palm',sep=',')
# np.array(jisi_palm).tofile(output_folder+'/jisi_palm',sep=',')
# np.array(times_v).tofile(output_folder+'/times_v',sep=',')
# np.array(jisi_v).tofile(output_folder+'/jisi_v',sep=',')
# np.array(times_n).tofile(output_folder+'/times_n',sep=',')
# np.array(jisi_n).tofile(output_folder+'/jisi_n',sep=',')

#================================================================================================
# TO BE MOVED TO A SEPARATE PROJECT
#================================================================================================


# K = 40
# folder_path = '../../SourceModeling/fMRI_data/'
# filenamebase = 'RegIVA-G_IVAGinit_AssistIVA-G_BAL98_pca_r1-'
# filename = folder_path + filenamebase + '1.mat'
# with h5py.File(filename, 'r') as data:
#     N,V = data['pcasig'][:].shape
#     # print('N = ', N)
    
# X = np.zeros((N,V,K))

# for k in tqdm(range(K)):
#     filename = folder_path + filenamebase + '{}.mat'.format(k+1)
#     with h5py.File(filename, 'r') as data:
#         # print(list(data.keys())) 
#         X[:,:,k] = data['pcasig'][:]
        
# W,C,_,_ = titan_iva_g_reg_numpy(X,nu=0,gamma_c=1.99)


# for k in range(K):
#     filename = folder_path+filenamebase+'{}'.format(k+1)
#     data = scipy.io.loadmat(filename)
#     print(data)

# import torch

# file_path = 'X_A_0.pt'  # Assurez-vous que le fichier est dans le même répertoire ou fournissez le chemin complet

# # Charger le fichier en spécifiant map_location pour le CPU
# X,A = torch.load(file_path, map_location=torch.device('cpu'))

# # Afficher le contenu
# print("Contenu du fichier :")
# print(X)
# print(A)
# # for key, value in data.items():
# #     print(f"{key}: {value.shape}, {value.dtype}")

# import pandas as pd
# import numpy as np


#-----------------------------------------------------------
# Chemin vers le fichier CSV
# csv_file = 'training_validation_losses_10_10_D3.csv'

# # Lire le fichier CSV avec pandas
# df = pd.read_csv(csv_file)

# # Convertir le DataFrame pandas en tableau NumPy
# tab = df.to_numpy()

# T = tab[:,0]
# train = tab[:,1]
# val = tab[:,2]

# plt.figure(figsize=(10, 6))

# # Tracer les données
# plt.plot(T, train, label='Train', marker='o', linestyle='-', color='b')
# plt.plot(T, val, label='Validation', marker='s', linestyle='--', color='r')

# # Ajouter un titre et des légendes
# plt.title('Loss (Train and Validation)', fontsize=24)
# plt.xlabel('Epochs', fontsize=20)
# plt.ylabel('Mean jISI', fontsize=20)
# plt.yscale('log')
# plt.legend( fontsize=18)

# # Ajouter une grille pour une meilleure lisibilité
# plt.grid(True)

# # Afficher la figure
# plt.show()

#-----------------------------------------------------------


# Charger le fichier sur le CPU
# X,A = torch.load('X_A_0.pt', map_location=torch.device('cpu'))
# X = X.float()

# Exemple d'utilisation des tenseurs
# Assurez-vous que toutes les variables utilisées dans les opérations sont de type Float
# A = A.float()
# print(A.dtype,X.dtype)
# print(A.shape,X.shape)
# W,_,_,times,jisi = titan_iva_g_reg_torch(X,nu=0,gamma_c=1.99,track_jisi=True,B=A)
# print(times[-1],jisi[-1])