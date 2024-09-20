import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
from class_exp import *
from class_algos import *
from algorithms.iva_g_numpy import *
from algorithms.titan_iva_g_reg_numpy import *
#------------------------------------------------------------------------------------------------------

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]
# identifiability_levels = [1e-2,1e-1,1])
# identifiability_levels_names = ['low identifiability','medium identifiability','high identifiability']
Ks = [5,10,20]
Ns = [10,20] 

common_parameters_1 = [Ks,Ns]
# common_parameters_2 = [[5,10,20],[5,10,20]]
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']

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



# algo_titan = TitanIvaG((0,0.4,0),name='titan',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_palm = TitanIvaG((0,0.4,0),name='palm',legend='PALM-IVA-G',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_iva_g_n = IvaG((0.5,0,0),name='iva_g_n',legend='IVA-G-N',crit_ext=1e-7,opt_approach='newton',library='numpy')
algo_iva_g_v = IvaG((0.5,1,0),name='iva_g_v',legend='IVA-G-V',crit_ext=1e-6,opt_approach='gradient',library='numpy')
# algo_palm_boost = TitanIvaG((0,0.4,0),name='palm_boost',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy',boost=True)


algos = [algo_palm,algo_iva_g_n,algo_iva_g_v]


N = 20
K = 20
T = 10000
rho_bounds = [0.2,0.3]
lambda_ = 0.25
epsilon = 1
X,A = generate_whitened_problem(T,K,N,epsilon,rho_bounds,lambda_)
Winit = make_A(K,N)
Cinit = make_Sigma(K,N,rank=K+10)

output_folder = 'Result_data/empirical convergence'
os.makedirs(output_folder,exist_ok=True)

_,_,_,times_palm,cost_palm,jisi_palm = titan_iva_g_reg_numpy(X.copy(),track_cost=True,track_jisi=True,gamma_c=1.99,B=A,max_iter_int=15,Winit=Winit.copy(),Cinit=Cinit)
for k in range(K):
    Winit[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(Winit[:, :, k] @ Winit[:, :, k].T), Winit[:, :, k])
_,_,_,jisi_n,times_n = iva_g_numpy(X.copy(),opt_approach='newton',A=A,W_init=Winit.copy(),W_diff_stop=1e-7)
_,_,_,jisi_v,times_v = iva_g_numpy(X.copy(),opt_approach='gradient',A=A,W_init=Winit.copy())

np.array(times_palm).tofile(output_folder+'/times_palm',sep=',')
np.array(cost_palm).tofile(output_folder+'/cost_palm',sep=',')
np.array(jisi_palm).tofile(output_folder+'/jisi_palm',sep=',')
np.array(times_v).tofile(output_folder+'/times_v',sep=',')
np.array(jisi_v).tofile(output_folder+'/jisi_v',sep=',')
np.array(times_n).tofile(output_folder+'/times_n',sep=',')
np.array(jisi_n).tofile(output_folder+'/jisi_n',sep=',')


# exp1 = ComparisonExperimentIvaG('titan vs palm',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
#                                 common_parameters_1,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=100,charts=False,legend=False)
# exp2 = ComparisonExperimentIvaG('palm part 2',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
                                # common_parameters_2,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=20,charts=False,legend=False)
# exp1.compute()
# exp2.get_data_from_folder('2024-05-16_02-22')
# exp1.make_table()

# exp2.make_charts(full=True)


# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp1.compute()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()
