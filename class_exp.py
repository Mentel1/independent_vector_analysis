import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
from algorithms.problem_simulation import *
import scipy.io

from class_algos import *
   
def generate_whitened_problem(T,K,N,epsilon=1,rho_bounds=[0,0],lambda_=0.95,rank=None): #, idx_W=None):
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
#On classe les résultats et les graphes dans une arborescence de 2 niveaux :
#un premier niveau de meta-paramètres qui dépendent du mode d'expérience 
#(donc un sous-dossier par combinaison de MP)
#puis un second niveau de paramètres commun (en l'occurrence K et N), c'est la que sont les graphes de comparaison
#Si on veut faire varier d'autres paramètres au niveau des algos, on définit plusieurs algorithmes séparés ! 


# L'idée de cette classe est de créer un objet "expérience" qui est déterminé par son nom 
# (lié au mode de l'expérience, mais pas que, à voir au cas par cas), par la date à laquelle
# elle est lancée, et qui contient/fabrique les résultats sous forme de données dans les algos 
# qu'elle implique ou dans des dossiers qui peuvent ou pas contenir des graphes. On veut pouvoir
# recréer un objet expérience à partir d'un dossier pour retravailler les données calculées et les présenter
# différemment par exemple
      
    def __init__(self,name,algos,meta_parameters,meta_parameters_titles,common_parameters,mode='multiparam',
                 T=10000,N_exp=100,table=False,table_fontsize=5,median=False,std=False,charts=False,legend=True,
                 legend_fontsize=5,title_fontsize=10,given_setup=False,store_setup=False):  
        self.algos = algos
        self.N_exp = N_exp
        self.mode = mode
        self.meta_parameters = meta_parameters
        self.meta_parameters_titles = meta_parameters_titles
        self.common_parameters = common_parameters
        # self.common_parameters_names = common_parameters_names
#parameters_name est une liste dont chaque élément est un tableau contenant les valeurs que prennent ces paramètres
        self.name = name
        self.T = T
        self.table = table
        self.table_fontsize = table_fontsize
        self.median = median
        self.std = std
        self.charts = charts
        self.legend = legend
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        now = datetime.now()
        self.date = now.strftime("%Y-%m-%d_%H-%M") 
        self.setup = {}
        self.store_setup = store_setup
        if given_setup:
            self.get_setup()
            self.store_setup = False
            
         
    def get_setup(self):
        pass
    
    def get_data_from_folder(self,date):
        self.date = date
        foldername = 'Result_data/' + self.date + ' ' + self.name
        Ks,Ns = self.common_parameters
        dimensions = (len(self.meta_parameters),len(Ks),len(Ns),self.N_exp)
        for algo in self.algos:
            algo.set_up_for_benchmark_experiment(dimensions)
            algo.fill_from_folder(foldername,self.meta_parameters,self.meta_parameters_titles,self.common_parameters,self.N_exp)

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
        output_folder = 'Result_data/' + self.date + ' ' + self.name   
        Ks,Ns = self.common_parameters
        n_cols = len(Ks)*len(Ns)
        best_results = self.best_perf(criterion='results')
        best_times = self.best_perf(criterion = 'times')
        tol_res,tol_time = tols
        # We consider that results_algo come from the same experiment
        filename = 'table results.txt' #+ algo.name + '.txt'
        output_path = os.path.join(output_folder, filename)
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
        output_folder = 'Result_data/' + self.date + ' ' + self.name
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    os.makedirs(output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    fig,ax = plt.subplots()
                    ax.set_xlabel('$T$ (s.)',fontsize=self.title_fontsize,labelpad=0)
                    ax.set_ylabel('$jISI$ score',fontsize=self.title_fontsize,labelpad=0)
                    for algo in self.algos:
                        ax.errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),
                                                yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),
                                                color=algo.color,label=algo.legend,elinewidth=2.5)
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
                    output_path = os.path.join(output_folder+'/charts/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K), filename)
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
                output_path = os.path.join(output_folder+'/charts/{}'.format(self.meta_parameters_titles[a]), filename)
                fig.savefig(output_path,dpi=200,bbox_inches='tight')
                plt.close(fig)
                                          
    def store_in_folder(self):
        output_folder = 'Result_data/' + self.date + ' ' + self.name
        os.makedirs(output_folder,exist_ok=True)
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    os.makedirs(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    if self.store_setup:
                        self.setup['Datasets {} {} {}'.format(a,K,N)].tofile(output_folder+'/{}/N = {} K = {}/Datasets'.format(self.meta_parameters_titles[a],N,K),sep=',')
                        self.setup['Mixing {} {} {}'.format(a,K,N)].tofile(output_folder+'/{}/N = {} K = {}/Mixing matrices'.format(self.meta_parameters_titles[a],N,K),sep=',')
                        self.setup['Winits {} {} {}'.format(a,K,N)].tofile(output_folder+'/{}/N = {} K = {}/Winits'.format(self.meta_parameters_titles[a],N,K),sep=',')
                        self.setup['Cinits {} {} {}'.format(a,K,N)].tofile(output_folder+'/{}/N = {} K = {}/Cinits'.format(self.meta_parameters_titles[a],N,K),sep=',')
                    for algo in self.algos: 
                        algo.results[a,ik,jn,:].tofile(output_folder+'/{}/N = {} K = {}/results_{}'.format(self.meta_parameters_titles[a],N,K,algo.name),sep=',')
                        algo.times[a,ik,jn,:].tofile(output_folder+'/{}/N = {} K = {}/times_{}'.format(self.meta_parameters_titles[a],N,K,algo.name),sep=',')
                        
        if self.charts:
            self.make_charts()
        if self.table:
            self.make_table()
                   
    def compute(self):
        Ks,Ns = self.common_parameters
        dimensions = (len(self.meta_parameters),len(Ks),len(Ns),self.N_exp)
        for algo in self.algos:
            algo.set_up_for_benchmark_experiment(dimensions)
        for a,metaparam in enumerate(self.meta_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    Datasets = np.zeros((self.N_exp,N,self.T,K))
                    Mixing = np.zeros((self.N_exp,N,N,K))
                    Winits = np.zeros((self.N_exp,N,N,K))
                    Cinits = np.zeros((self.N_exp,K,K,N))
                    for exp in range(self.N_exp):
                        if self.mode == 'identifiability':
                            X,A = generate_whitened_problem(self.T,K,N,epsilon=metaparam)
                        elif self.mode == 'multiparam':
                            rho_bounds,lambda_ = metaparam
                            X,A = generate_whitened_problem(self.T,K,N,rho_bounds=rho_bounds,lambda_=lambda_)
                        elif self.mode == 'effective rank':
                            X,A = generate_whitened_problem(self.T,K,N,rank=metaparam)
                        Datasets[exp,:,:,:] = X
                        Mixing[exp,:,:,:] = A
                        Winits[exp,:,:,:] = make_A(K,N)
                        Cinits[exp,:,:,:] = make_Sigma(K,N,rank=K+10) #do we bring this rank into question ?
                        for algo in self.algos:
                            algo.fill_experiment(X,A,(a,ik,jn,exp),Winits[exp,:,:,:].copy(),Cinits[exp,:,:,:].copy())
                            print(a,' K =',K,' N =',N,algo.name,' : ',algo.results[a,ik,jn,exp],algo.times[a,ik,jn,exp])
                    self.setup['Datasets {} {} {}'.format(a,K,N)] = Datasets
                    self.setup['Mixing {} {} {}'.format(a,K,N)] = Mixing
                    self.setup['Winits {} {} {}'.format(a,K,N)] = Winits
                    self.setup['Cinits {} {} {}'.format(a,K,N)] = Cinits
        self.store_in_folder()

    def draw_jisi_evolutions(self):
        output_folder = self.date + ' ' + self.name
        os.makedirs(output_folder,exist_ok=True)
        Ks,Ns = self.common_parameters
        for a,metaparam in enumerate(self.meta_parameters):
            fig_global,axes = plt.subplots(len(Ks),len(Ns))
            fig_global.supxlabel('Iteration (external loop)',fontsize=self.title_fontsize)
            fig_global.supylabel('jISI score',fontsize=self.title_fontsize)
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    if ik == 0:
                        axes[ik,jn].set_title('N = {}'.format(N),fontsize=self.title_fontsize)
                    if jn == 0:
                        axes[ik,jn].set_ylabel('K = {}'.format(K),fontsize=self.title_fontsize)
                    os.makedirs(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K))
                    fig,ax = plt.subplots()
                    fig.supxlabel('Iteration (external loop)',fontsize=self.title_fontsize)
                    fig.supylabel('jISI score',fontsize=self.title_fontsize)
                    ax.set_yscale('log')
                    axes[ik,jn].set_yscale('log')
                    X,A = generate_whitened_problem(self.T,metaparam,K,N,mode=self.mode)
                    Winit = make_A(K,N)
                    Cinit = make_Sigma(K,N)
                    for algo in self.algos:
                        t = -time()
                        jisi = algo.solve_with_jisi(self,X,A,Winit,Cinit)
                        t += time()
                        res = jisi[-1]
                        ax.plot(jisi,color=algo.color,label=algo.legend +' time = {:.2E}, jISI = {:.3f}'.format(t,res),linewidth=0.5)
                        axes[ik,jn].plot(jisi,color=algo.color,label=algo.legend +' time = {:.2E}, jISI = {:.3f}'.format(t,res),linewidth=0.5)
                    ax.legend(loc=1,fontsize=self.legend_fontsize)
                    for extension in ['.eps','.png']:
                        filename = 'jisi evolutions' + extension
                        output_path = os.path.join(output_folder+'/{}/N = {} K = {}'.format(self.meta_parameters_titles[a],N,K), filename)
                        fig.savefig(output_path,dpi=200)
            fig_global.subplots_adjust(wspace=0.2,hspace=0.4)
            fig_global.legend(loc=1,fontsize=self.legend_fontsize)
            fig_global.savefig(output_path,dpi=200)     


class ApplicationToRealData:
    
    def __init__(self,data_path,algos):
        
        self.algos = algos
        self.data_path = data_path
        self.data = scipy.io.loadmat(data_path)
        self.X =  None #A changer 
        
    def define_order(self,method):
        pass
    
    def solve(self):
        for algo in self.algos:
            W,_,_,_ = algo.solve(self.X,Winit=None,C_init=None)
        
        
    






                    

        

