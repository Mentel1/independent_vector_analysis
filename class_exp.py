import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
from algorithms.problem_simulation import *
import scipy.io

from class_algos import *
   
def generate_whitened_problem(T,K,N,mode='multiparam',metaparam=None):
    epsilon=1
    rho_bounds=[0.4,0.6]
    lambda_=0.5
    rank=None
    if metaparam == None:
        raise ValueError('metaparam must be specified')
    if mode == 'identifiability':
        epsilon = metaparam
    elif mode == 'multiparam':
        rho_bounds,lambda_ = metaparam
    elif mode == 'effective rank':
        rank = metaparam
    else:
        raise ValueError('Unknown mode {}'.format(mode))
    A = make_A(K,N)
    # A = full_to_blocks(A,idx_W,K)
    if rank == None:
        rank = K+10
    Sigma = make_Sigma(K,N,rank=rank,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
    S = make_S(Sigma,T)
    X = make_X(S,A)
    X_,U = whiten_data_numpy(X)
    A_ = np.einsum('nNk,Nvk->nvk',U,A)
    return X_,A_

class ComparisonExperimentIvaG:
#On classe les résultats et les graphes dans une arborescence de 2 niveaux : un premier niveau de meta-paramètres qui dépendent du mode d'expérience (donc un sous-dossier par combinaison de MP) puis un second niveau de paramètres commun (en l'occurrence K et N), c'est la que sont les graphes de comparaison.
#Si on veut faire varier d'autres paramètres au niveau des algos, on définit plusieurs algorithmes séparés ! 

# L'idée de cette classe est de créer un objet "expérience" qui est déterminé par son nom (lié au mode de l'expérience, mais pas que, à voir au cas par cas), par la date à laquelle elle est lancée, et qui contient/fabrique les résultats sous forme de données dans les algos qu'elle implique ou dans des dossiers qui peuvent ou pas contenir des graphes. On veut pouvoir recréer un objet expérience à partir d'un dossier pour retravailler les données calculées et les présenter différemment par exemple
      
    def __init__(self,name,meta_parameters,meta_parameters_titles,common_parameters,algos,mode='multiparam',date=None,T=10000,N_exp=100,table_fontsize=5,median=False,std=False,legend=True,legend_fontsize=5,title_fontsize=10):  
        self.algos = algos
        self.N_exp = N_exp
        self.mode = mode
        self.meta_parameters = meta_parameters
        self.meta_parameters_titles = meta_parameters_titles
        self.common_parameters = common_parameters
        self.name = name
        if date:
            self.date = date
            self.exists_setup = True
        else:
            now = datetime.now()
            self.date = now.strftime("%Y-%m-%d_%H-%M")
            self.exists_setup = False
        self.output_folder = 'Result_data/' + self.date + '_' + self.name
        self.T = T
        self.table_fontsize = table_fontsize
        self.median = median
        self.std = std
        self.legend = legend
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize 
        self.setup = {}
    
    def get_results_from_folder(self,N,K,a):    
        output_path_individual = os.path.join(self.output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
        for algo in self.algos:
            algo.fill_from_folder(output_path_individual)
    
    def get_setup_from_folder(self,N,K,a):
        output_path_individual = os.path.join(self.output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
        for setup_var in self.setup.keys():
            var_path = os.path.join(output_path_individual,setup_var)
            self.setup[setup_var].fromfile(var_path,sep=',')
            
    def set_algos(self,new_algos):
        self.algos = new_algos

    def best_perf(self,criterion='results'):
        Ks,Ns = self.common_parameters
        best_perfs = np.zeros((len(self.meta_parameters),len(Ks),len(Ns)))
        for a,meta_param in enumerate(self.meta_parameters):
                for ik,K in enumerate(Ks):
                    for jn,N in enumerate(Ns):
                        if criterion == 'results':
                            perfs = [np.mean(algo.results[a,ik,jn,:]) for algo in self.algos]
                        else:
                            perfs = [np.mean(algo.times[a,ik,jn,:]) for algo in self.algos]
                        best_perfs[a,ik,jn] = min(perfs)
        return best_perfs
   
    def make_table(self,tols=(1e-4,1e-2)): 
        Ks,Ns = self.common_parameters
        n_cols = len(Ks)*len(Ns)
        best_results = self.best_perf(criterion='results')
        best_times = self.best_perf(criterion = 'times')
        tol_res,tol_time = tols
        # We consider that results_algo come from the same experiment
        filename = 'table results.txt' #+ algo.name + '.txt'
        output_path = os.path.join(self.output_folder, filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as file:
            file.write('\\begin{table}[h!]\n\\caption{'+'blablabla'+'}\n\\vspace{0.4cm}\n')
            file.write('\\fontsize{{{}pt}}{{{}pt}}\selectfont\n'.format(self.table_fontsize,self.table_fontsize))
            file.write('\\begin{{tabular}}{{{}}}\n'.format('cm{0.5cm}m{0.5cm}'+n_cols*'c'))
            file.write('& &')
            for K in Ks:
                file.write(' & \\multicolumn{{{}}}{{c}}{{$K$ = {}}}'.format(len(Ns),K))
            file.write('\\\\\n')
            for ik,K in enumerate(Ks):
                file.write(' \\cmidrule(lr){{{}-{}}}'.format(4+ik*len(Ns),3+(ik+1)*len(Ns)))
            file.write('\n')
            file.write('& &')
            for K in Ks:
                for N in Ns:
                    file.write(' & $N$ = {}'.format(N))
            file.write('\\\\\n')
            for algo_index,algo in enumerate(self.algos):
                file.write('\\midrule\n')
                file.write('\\multirow{{{}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\small{{\\textbf{{{}}}}}}}}}'.format(3*len(self.meta_parameters),algo.legend))
                for a,metaparam in enumerate(self.meta_parameters):
                    file.write('& \\multirow{{{}}}{{*}}{{\\begin{{tabular}}{{c}} {} \\end{{tabular}}}}& $\\mu_{{\\rm jISI}}$'.format(2+self.std+2*self.median,self.meta_parameters_titles[a]))
                    for ik,K in enumerate(Ks):
                        for jn,N in enumerate(Ns):
                            if np.mean(algo.results[a,ik,jn,:]) <= best_results[a,ik,jn] + tol_res:
                                file.write(' & \\textbf{{{:.2E}}}'.format(np.mean(algo.results[a,ik,jn,:])))
                            else:
                                file.write(' & {:.2E}'.format(np.mean(algo.results[a,ik,jn,:])))
                    file.write('\\\\\n')
                    if self.median:
                        file.write('& & $\\widehat{\\mu}_{\\rm jISI}$')
                        for ik,K in enumerate(Ks):
                            for jn,N in enumerate(Ns):
                                file.write(' & {:.2E}'.format(np.median(algo.results[a,ik,jn,:])))
                        file.write('\\\\\n')
                    if self.std:
                        file.write('& & $\\sigma_{\\rm jISI}$')
                        for ik,K in enumerate(Ks):
                            for jn,N in enumerate(Ns):
                                file.write(' & {:.2E}'.format(np.std(algo.results[a,ik,jn,:])))
                        file.write('\\\\\n')
                    if self.median:
                        file.write('& & $\\widehat{\\sigma}_{\\rm jISI}$')
                        for ik,K in enumerate(Ks):
                            for jn,N in enumerate(Ns):
                                file.write(' & {:.2E}'.format(np.median(np.abs(algo.results[a,ik,jn,:]-np.mean(algo.results[a,ik,jn,:])))))
                        file.write('\\\\\n')
                    file.write('& & $\\mu_T$')
                    for ik,K in enumerate(Ks):
                        for jn,N in enumerate(Ns):
                            if np.mean(algo.times[a,ik,jn,:]) <= best_times[a,ik,jn] + tol_time:
                                file.write(' & \\textit{{\\textbf{{{:.1f}}}}}'.format(np.mean(algo.times[a,ik,jn,:])))
                            else:
                                file.write(' & {:.1f}'.format(np.mean(algo.times[a,ik,jn,:])))
                    file.write('\\\\\n')
                    if a == len(self.meta_parameters)-1:
                        file.write('\\bottomrule\n')
                        file.write('\\\\\n')
                    else:
                        file.write('\\cmidrule(lr){{2-{}}}'.format(3+n_cols))
            file.write('\\end{tabular}\n\\end{table}')

    def make_charts(self,full=False):
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    os.makedirs(self.output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    fig,ax = plt.subplots()
                    ax.set_xlabel('$T$ (s.)',fontsize=self.title_fontsize,labelpad=0)
                    ax.set_ylabel('$jISI$ score',fontsize=self.title_fontsize,labelpad=0)
                    for algo in self.algos:
                        ax.errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),color=algo.color,label=algo.legend,elinewidth=2.5)
                    ax.set_yscale('log')
                    ax.grid(which='both')
                    # yticks = ax.get_yticks(minor=True)
                    # print(yticks)
                    # yticklabels = ['{:.0e}'.format(tick) for tick in yticks]
                    # ax.set_yticklabels(yticklabels)
                    # xticks = ax.get_xticks()
                    # xticklabels = ['{:.0e}'.format(tick) for tick in xticks]
                    # ax.set_xticklabels(xticklabels)
                    if self.legend:
                        fig.legend(loc=2,fontsize=self.legend_fontsize)
                    filename = 'comparison {} N = {} K = {}'.format(self.meta_parameters_titles[a],N,K)
                    output_path = os.path.join(self.output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K), filename)
                    fig.savefig(output_path,dpi=200,bbox_inches='tight')
                    plt.close(fig)
            if full:
                fig,ax = plt.subplots(len(Ns),len(Ks),figsize=(12, 8))
                fig.text(0.5, 0.04, '$T$ (s.)', ha='center', fontsize=self.title_fontsize)
                fig.text(0.04, 0.5, '$jISI$ score', va='center', rotation='vertical', fontsize=self.title_fontsize)
                plt.yscale('log')
                for ik,K in enumerate(Ks):
                    for jn,N in enumerate(Ns):
                        if ik == 0:
                            ax[jn,ik].set_title('N = {}'.format(N))
                        for algo in self.algos:
                            ax[jn,ik].errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),
                                                yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),
                                                color=algo.color,label=algo.legend,elinewidth=2.5)
                if self.legend:
                    fig.legend(loc=2,fontsize=self.legend_fontsize)
                filename = 'comparison {}.png'.format(self.meta_parameters_titles[a])
                output_path = os.path.join(self.output_folder+'/charts/{}'.format(self.meta_parameters_titles[a]), filename)
                fig.savefig(output_path,dpi=200,bbox_inches='tight')
                plt.close(fig)
                                          
    def store_in_folder(self,N,K,a):
        output_path_individual = self.output_folder+ f'/{self.meta_parameters_titles[a]}/N = {N} K = {K}'
        os.makedirs(output_path_individual,exist_ok=True)
        if not self.exists_setup:
            for setup_var in self.setup.keys():
                var_path = os.path.join(output_path_individual,setup_var)
                self.setup[setup_var].tofile(var_path,sep=',')       
        for algo in self.algos:
            for result in algo.results.keys():
                res_path = os.path.join(output_path_individual,algo.name + '_' + result)
                algo.results[result].tofile(res_path,sep=',')      
                   
    def compute_multi_runs(self):
        Ks,Ns = self.common_parameters
        for algo in self.algos:
            algo.results['total_times'] = np.zeros(self.N_exp)
            algo.results['final_jisi'] = np.zeros(self.N_exp)
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    if self.exists_setup:
                        self.get_setup_from_folder(N,K,a)
                    else:
                        self.create_setup_multi_runs(metaparam, K, N) 
                    for exp in range(self.N_exp):
                        for algo in self.algos:
                            algo.fill_experiment(self.setup['Datasets'][exp,:,:,:],self.setup['Mixings'][exp,:,:,:],exp,self.setup['Winits'][exp,:,:,:],self.setup['Cinits'][exp,:,:,:])
                            print(a,' K =',K,' N =',N,algo.name,' : ',algo.results['final_jisi'][exp],algo.results['total_times'][exp])
                self.store_in_folder(N,K,a)

    def create_setup_multi_runs(self,metaparam,K,N):
        self.setup['Datasets'] = np.zeros((self.N_exp,N,self.T,K))
        self.setup['Mixings'] = np.zeros((self.N_exp,N,N,K))
        self.setup['Winits'] = np.zeros((self.N_exp,N,N,K))
        self.setup['Cinits'] = np.zeros((self.N_exp,K,K,N))
        for exp in range(self.N_exp):
            X,A = generate_whitened_problem(self.T,K,N,mode=self.mode,metaparam=metaparam)
            self.setup['Datasets'][exp,:,:,:] = X
            self.setup['Mixings'][exp,:,:,:] = A
            self.setup['Winits'][exp,:,:,:] = make_A(K,N)
            self.setup['Cinits'][exp,:,:,:] = make_Sigma(K,N,rank=K+10)
            
    def compute_empirical_convergence(self,detailed=False):
        res_types = ['jisi','costs','times']
        if detailed:
            res_types.append('detailed_times','detailed_costs','detailed_jisi')
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    if self.exists_setup:
                        self.get_setup_from_folder()
                    else:
                        self.create_setup_empirical_convergence(metaparam,K,N)
                    for algo in self.algos:
                        res = algo.solve(self.setup['Dataset'],self.setup['Winit'],self.setup['Cinit'],B=self.setup['Mixing'],track_jisi=True,track_costs=True,track_times=True)
                        for res_type in res_types:
                            algo.results[res_type] = res[res_type]
                    self.store_in_folder(N,K,a)
       
    def create_setup_empirical_convergence(self,metaparam,K,N):
        self.setup['Dataset'],self.setup['Mixing'] = generate_whitened_problem(self.T,K,N,mode=self.mode,metaparam=metaparam)
        self.setup['Winit'] = make_A(K,N)
        self.setup['Cinit'] = make_Sigma(K,N,rank=K+10)
        
           
    def draw_empirical_convergence(self):
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    output_path_individual = self.output_folder+ f'/{self.meta_parameters_titles[a]}/N = {N} K = {K}'
                    for algo in self.algos:
                        algo.fill_from_folder(output_path_individual)
                    res_types = self.algos[0].results.keys() & {'costs','jisi','detailed_costs','detailed_jisi'}
                    for res_type in res_types:
                        if 'detailed' in res_type:
                            times = algo.results['detailed_times']
                        else:
                            times = algo.results['times']
                        fig,ax = plt.subplots()
                        ax.set_yscale('log')
                        for algo in self.algos:                        
                            ax.plot(algo.results[res_type],color=algo.color,label=algo.legend,linewidth=0.5)
                            ax.legend(loc=1,fontsize=self.legend_fontsize)
                            for extension in ['.eps','.png']:
                                fig_path = os.path.join(output_path_individual, res_type, extension)
                                fig.savefig(fig_path,dpi=200)
                            ax.plot(algo.results[res_type],times,color=algo.color,label=algo.legend,linewidth=0.5)
                            ax.legend(loc=1,fontsize=self.legend_fontsize)
                            for extension in ['.eps','.png']:
                                fig_path = os.path.join(output_path_individual, res_type, '_times', extension)
                                fig.savefig(fig_path,dpi=200)






                    

        

