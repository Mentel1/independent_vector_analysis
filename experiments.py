import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
from class_exp import *
from class_algos import *
#------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1] #,rho_bounds_2]
lambdas = [lambda_1] #,lambda_2]
# identifiability_levels = [1e-2,1e-1,1])
# identifiability_levels_names = ['low identifiability','medium identifiability','high identifiability']
Ks = [10,20,50]
Ns = [10,20] #,10]

common_parameters = [Ks,Ns]
metaparameters_titles_multiparam = ['Case A'] #,'Case B','Case C','Case D']

def get_metaparameters(rhos,lambdas):
    metaparameters_multiparam = []
    for rho_bounds in rhos:
        for lambda_ in lambdas:
            metaparameters_multiparam.append((rho_bounds,lambda_))
    return metaparameters_multiparam
         
# ------------------------------------------------------------------------------------------------------------------------------                
# ------------------------------------------------------------------------------------------------------------------------------------------

label_size = 60
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
plt.rcParams['text.usetex'] = True

metaparameters_multiparam = get_metaparameters(rhos,lambdas)


algo_titan1 = TitanIvaG((0,0.4,0),name='titan1',alpha=0.1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_titan2 = TitanIvaG((0,0.4,0),name='titan2',alpha=0.5,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_titan3 = TitanIvaG((0,0.4,0),name='titan3',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_titan4 = TitanIvaG((0,0.4,0),name='titan4',alpha=5,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_titan5 = TitanIvaG((0,0.4,0),name='titan5',alpha=10,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
# algo_titan_torch = TitanIvaG((0,0.8,0),name='titan_torch',gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='torch')
# algo_iva_g_n_numpy = IvaG((0.5,0,0),name='iva_g_n_numpy',legend='NN',crit_ext=1e-7,opt_approach='newton',library='numpy')
# algo_iva_g_n_torch = IvaG((1,0,0),name='iva_g_n_torch',legend='NT',crit_ext=1e-7,opt_approach='newton',library='torch')
# algo_iva_g_v_numpy = IvaG((0.5,1,0),name='iva_g_v_numpy',legend='VN',crit_ext=1e-6,opt_approach='gradient',library='numpy')
# algo_iva_g_v_torch = IvaG((1,1,0),name='iva_g_v_torch',legend='VT',crit_ext=1e-6,opt_approach='gradient',library='torch')
# algo_fast_iva_g_v_torch = IvaG((1,1,0.5),name='fast_iva_g_v_torch',legend='FVT',crit_ext=1e-6,opt_approach='gradient',library='torch',fast=True)
# algo_fast_iva_g_n_torch = IvaG((1,0,0.5),name='fast_iva_g_n_torch',legend='FNT',crit_ext=1e-7,opt_approach='newton',library='torch',fast=True)
# algo_fast_iva_g_v_numpy = IvaG((0.5,1,0.5),name='fast_iva_g_v_numpy',legend='FVN',crit_ext=1e-6,opt_approach='gradient',library='numpy',fast=True)
# algo_fast_iva_g_n_numpy = IvaG((0.5,0,0.5),name='fast_iva_g_n_numpy',legend='FNN',crit_ext=1e-7,opt_approach='newton',library='numpy',fast=True)


algos = [algo_titan1,algo_titan2,algo_titan3,algo_titan4,algo_titan5] #algo_fast_iva_g_v_torch,algo_fast_iva_g_n_numpy,algo_fast_iva_g_n_torch] #,algo_iva_g_v_numpy,algo_iva_g_v_torch,algo_iva_g_n_torch,algo_iva_g_n_numpy] #,algo_iva_g_v_torch] #,algo_titan_numpy,algo_titan_torch,algo_iva_g_v_numpy,algo_iva_g_n_numpy]

exp2 = ComparisonExperimentIvaG('alpha benchmark',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
                                common_parameters,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=50,charts=False,legend=False)
exp2.compute()
# exp2.get_data_from_folder('2024-05-16_02-22')
exp2.make_table()
# exp2.make_charts(full=True)


# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp2.compute()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()
