import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from tqdm import tqdm
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
effective_ranks = [None]
Ks = [5,10] #,15,20,30,40,50] #,20]
Ns = [10,20] 

common_parameters_1 = [Ks,Ns]
# common_parameters_2 = [[5,10,20],[5,10,20]]
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D','Case E','Case F','Case G','Case H']

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
metaparameters_multiparam = effective_ranks



# algo_titan = TitanIvaG((0,0.4,0),name='titan',alpha=1,gamma_w=0.99,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_palm_np = TitanIvaG((0,0.4,0),name='palm_np',legend='PALM-IVA-G-NP',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy')
algo_palm_torch = TitanIvaG((0,0.8,0),name='palm_torch',legend='PALM-IVA-G-TORCH',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='torch')
# algo_iva_g_n = IvaG((0.5,0,0),name='iva_g_n',legend='IVA-G-N',crit_ext=1e-7,opt_approach='newton',library='numpy')
# algo_iva_g_v = IvaG((0.5,1,0),name='iva_g_v',legend='IVA-G-V',crit_ext=1e-6,opt_approach='gradient',library='numpy')
# algo_palm_boost = TitanIvaG((0,0.4,0),name='palm_boost',alpha=1,gamma_w=0.99,gamma_c = 1.99,nu=0,crit_ext=1e-10,crit_int=1e-10,library='numpy',boost=True)


algos = [algo_palm_np,algo_palm_torch] #,algo_iva_g_v] #algo_iva_g_n,


exp1 = ComparisonExperimentIvaG('rank effect',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
                                common_parameters_1,'effective rank',title_fontsize=50,legend_fontsize=6,N_exp=3,charts=False,legend=False)
# exp2 = ComparisonExperimentIvaG('palm part 2',algos,metaparameters_multiparam,metaparameters_titles_multiparam,
                                # common_parameters_2,'multiparam',title_fontsize=50,legend_fontsize=6,N_exp=20,charts=False,legend=False)
exp1.compute()
# exp2.get_data_from_folder('2024-05-16_02-22')
# exp1.make_table()

# exp2.make_charts(full=True)

#=====================================================================================================================================================

# if __name__ == '__main__':
#     import cProfile, pstats
#     profiler = cProfile.Profile()
#     profiler.enable()
#     exp1.compute()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

#=====================================================================================================================================================

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

#=====================================================================================================================================================

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
