import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
from algorithms.problem_simulation import *
import scipy.io

from class_algos import *
   
class ComparisonExperimentIvaG:
#On classe les résultats et les graphes dans une arborescence de 2 niveaux : un premier niveau de data-paramètres qui dépendent du mode d'expérience (donc un sous-dossier par combinaison de MP) puis un second niveau de paramètres commun (en l'occurrence K et N), c'est la que sont les graphes de comparaison.
#Si on veut faire varier d'autres paramètres au niveau des algos, on définit plusieurs algorithmes séparés ! 

# L'idée de cette classe est de créer un objet "expérience" qui est déterminé par son nom (lié au mode de l'expérience, mais pas que, à voir au cas par cas), par la date à laquelle elle est lancée, et qui contient/fabrique les résultats sous forme de données dans les algos qu'elle implique ou dans des dossiers qui peuvent ou pas contenir des graphes. On veut pouvoir recréer un objet expérience à partir d'un dossier pour retravailler les données calculées et les présenter différemment par exemple
      
    def __init__(self,name,data_parameters,data_parameters_titles,common_parameters,algos,mode='multiparam',date=None,T=10000,N_exp=100,table_fontsize=8,median=False,std=False,updates=False,legend=True,legend_fontsize=10,title_fontsize=10):  
        self.algos = algos
        self.N_exp = N_exp
        self.mode = mode
        self.data_parameters = data_parameters
        self.data_parameters_titles = data_parameters_titles
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
        self.updates = updates
        self.legend = legend
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize 
        self.setup = {}
    
    
    def to_dict(self):
        algo_names = [algo.name for algo in self.algos]
        return {'N_exp':self.N_exp,'name':self.name,'common_parameters':self.common_parameters,'data_parameters':self.data_parameters,'data_parameters_titles':self.data_parameters_titles,'mode':self.mode,'T':self.T,'date':self.date,'table_fontsize':self.table_fontsize,'median':self.median,'std':self.std,'legend':self.legend,'title_fontsize':self.title_fontsize,'legend_fontsize':self.legend_fontsize,'algo_names':algo_names}
        
        
    def save(self):
        config = self.to_dict()
        algo_folder = self.output_folder + '/algos'
        os.makedirs(algo_folder,exist_ok=True)
        filepath = self.output_folder + '/config.json' 
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)      
        for algo in self.algos:
            algo_config = algo.to_dict()
            algo_path = algo_folder + '/' + algo.name
            with open(algo_path, 'w') as f:
                json.dump(algo_config, f, indent=2)
        
    
    @classmethod
    def from_folder(cls,folderpath):
        filepath = 'Result_data/' + folderpath + '/config.json'
        with open(filepath, 'r') as f:
            config = json.load(f)
        algos = []
        algo_names = config.pop('algo_names')
        for algo_name in algo_names:
            algopath = 'Result_data/' + folderpath + '/algos/' + algo_name
            with open(algopath, 'r') as f:
                algo_config = json.load(f)
            algo = IvaGAlgorithms.from_dict(algo_config)
            algos.append(algo)
        config['algos'] = algos
        return ComparisonExperimentIvaG(**config)
    
    
    def get_results_from_folder(self,param_path):    
        output_path_results = f'/results/{param_path}'
        for algo in self.algos:
            algo.fill_from_folder(output_path_results)
    
    def get_data_from_folder(self,param_path):
        output_path_data = f'/data/{param_path}'
        for setup_var in self.setup.keys():
            var_path = os.path.join(output_path_data,setup_var)
            self.setup[setup_var].fromfile(var_path,sep=',')
            
    def set_algos(self,new_algos):
        self.algos = new_algos

    def compute_features(self,algo,Ks,Ns):
        algo.results['full_results_jisi'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        algo.results['full_results_times'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        if self.updates:
            algo.results['full_results_updates'] = np.zeros((len(self.data_parameters),len(Ks),len(Ns),self.N_exp))
        for a,dataparam in enumerate(self.data_parameters_titles):
            for jn,N in enumerate(Ns):
                for ik,K in enumerate(Ks):
                    path = self.output_folder+f'/{dataparam}/N_{N}_K_{K}'
                    algo.fill_from_folder(path)
                    algo.results['full_results_jisi'][a,ik,jn,:] = algo.results['final_jisi']
                    algo.results['full_results_times'][a,ik,jn,:] = algo.results['total_times']
                    if self.updates:
                        algo.results['full_results_updates'][a,ik,jn,:] = algo.results['number_updates']
        algo.results['mean_jisi'] = np.mean(algo.results['full_results_jisi'],axis=-1)
        algo.results['mean_times'] = np.mean(algo.results['full_results_times'],axis=-1)
        if self.updates:
            algo.results['mean_updates'] = np.mean(algo.results['full_results_updates'],axis=-1)
        print(algo.results['mean_jisi'])
        if self.std:
            algo.results['std_jisi'] = np.std(algo.results['full_results_jisi'],axis=-1)
            algo.results['std_times'] = np.std(algo.results['full_results_times'],axis=-1)
            if self.updates:
                algo.results['std_updates'] = np.std(algo.results['full_results_updates'],axis=-1)
        if self.median:
            algo.results['median_jisi'] = np.median(algo.results['full_results_jisi'],axis=-1)
            algo.results['median_dev_jisi'] = np.median(abs(algo.results['full_results_jisi'] - algo.results['median_jisi']),axis=-1)
            algo.results['median_times'] = np.median(algo.results['full_results_times'],axis=-1)
            algo.results['median_dev_times'] = np.median(abs(algo.results['full_results_times'] - algo.results['median_times']),axis=-1)
            if self.updates:
                algo.results['median_updates'] = np.median(algo.results['full_results_updates'],axis=-1)
                algo.results['median_updates'] = np.median(abs(algo.results['full_results_updates'] - algo.results['median_updates']),axis=-1)
    
    def list_features(self):
        res = ['mean_jisi','mean_times']
        if self.updates:
            res += ['mean_updates']
            if self.std:
                res += ['std_updates']
            if self.median:
                res += ['median_updates','median_dev_updates']
        if self.std:
            res += ['std_jisi','std_times']
        if self.median:
            res += ['median_jisi','median_dev_jisi','median_times','median_dev_times']
        return res
        
    def best_perf(self,feature):
        all_perfs = np.array([algo.results[feature] for algo in self.algos])
        return np.min(all_perfs, axis=0)
   
    base_feature_names = {'mean_jisi':'$\\mu_{\\rm jISI}$','mean_times':'$\mu_\\texttt{T}$','mean_updates':'$\mu_\\texttt{N}$','median_jisi':'$\\widehat{\\mu}_{\\rm jISI}$','std_jisi':'$\\sigma_{\\rm jISI}$','median_dev_jisi':'$\\widehat{\\sigma}_{\\rm jISI}$',}
    base_tols = {'mean_jisi':1e-4,'mean_times':1e-2}
    
    def make_table(self,tols=base_tols,feature_names=base_feature_names,filename='table_results.txt'):
        Ks,Ns = self.common_parameters
        for algo in self.algos:
            self.compute_features(algo,Ks,Ns)  
        features = self.list_features()
        n_cols = len(Ks)*len(Ns)
        bold_numbers = {}
        for feature in ['mean_jisi','mean_times']:
            best_feature = self.best_perf(feature)
            for algo in self.algos:
                bold_numbers[(feature,algo)] = algo.results[feature] <= best_feature + tols[feature]
        # We consider that results_algo come from the same experiment 
        output_path = os.path.join(self.output_folder, filename)
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, 'a') as file:
            file.write('\\begin{table}[h!]\n\\caption{'+'blablabla'+'}\n\\vspace{0.4cm}\n')
            file.write(f'\\fontsize{{{self.table_fontsize}pt}}{{{self.table_fontsize}pt}}\selectfont\n')
            file.write('\\begin{tabular}{cm{1cm}m{0.5cm}'+n_cols*'c'+'}\n')
            file.write('& &')
            for K in Ks:
                file.write(f' & \\multicolumn{{{len(Ns)}}}{{c}}{{$K$ = {K}}}')
            file.write('\\\\\n')
            for ik,K in enumerate(Ks):
                file.write(f' \\cmidrule(lr){{{4+ik*len(Ns)}-{3+(ik+1)*len(Ns)}}}')
            file.write('\n')
            file.write('& &')
            for K in Ks:
                for N in Ns:
                    file.write(f' & $N$ = {N}')
            file.write('\\\\\n')
            for algo in self.algos:
                self.write_algo_in_table(file,algo,Ks,Ns,n_cols,features,feature_names,bold_numbers)
            file.write('\\end{tabular}\n\\end{table}')

    def write_algo_in_table(self,file,algo,Ks,Ns,n_cols,features,feature_names,bold_numbers):
        file.write('\\midrule\n')
        file.write(f'\\multirow{{{3*len(self.data_parameters)}}}{{*}}{{\\raisebox{{-2\\height}}{{\\rotatebox[origin=c]{{90}}{{\\makebox[0pt][c]{{\\Large{{\\textbf{{{algo.legend}}}}}}}}}}}}}')
        for a,dataparam_title in enumerate(self.data_parameters_titles):
            file.write(f'& \\multirow{{{len(features)}}}{{*}}{{\\begin{{tabular}}{{c}} {dataparam_title} \\end{{tabular}}}}')
            for idx,feature in enumerate(features):
                if idx > 0:
                    file.write('& ')
                file.write('& ' + feature_names[feature])
                for ik,_ in enumerate(Ks):
                    for jn,_ in enumerate(Ns):
                        value = algo.results[feature][a,ik,jn]
                        bold = False
                        exp_notation = True
                        if feature in ['mean_jisi','mean_times']:
                            bold = bold_numbers[(feature,algo)][a,ik,jn]
                        if feature in ['mean_updates','mean_times']:
                            exp_notation = False
                        self.write_in_table(file,value,bold,exp_notation)
                file.write('\\\\\n')
            if a == len(self.data_parameters)-1:
                file.write('\\bottomrule\n')
                file.write('\\\\\n')
            else:
                file.write(f'\\cmidrule(lr){{2-{3+n_cols}}}')

    def write_in_table(self,file,value,bold=False,exp_notation=True):
        fmt = '.2E' if exp_notation else '.1f'
        if bold:
            file.write(f' & \\textbf{{{value:{fmt}}}}')
        else:
            file.write(f' & {value:{fmt}}')
                                          
    def store_in_folder(self,param_path,subfolder='data'):
            full_path = f'{self.output_folder}/{subfolder}/{param_path}'
            os.makedirs(full_path,exist_ok=True)
            if subfolder == 'data':
                for setup_var in self.setup.keys():
                    var_path = os.path.join(full_path,setup_var)
                    self.setup[setup_var].tofile(var_path,sep=',')
            else:     
                for algo in self.algos:
                    for res_var in algo.results.keys():
                        res_path = os.path.join(full_path,algo.name + '_' + res_var)
                        algo.results[res_var].tofile(res_path,sep=',')      
                   
    def compute_multi_runs(self):
        Ks,Ns = self.common_parameters
        for algo in self.algos:
            algo.results['total_times'] = np.zeros(self.N_exp)
            algo.results['final_jisi'] = np.zeros(self.N_exp)
            if (self.updates):
                algo.results['number_updates'] = np.zeros(self.N_exp)      
        for a,dataparam in enumerate(self.data_parameters):
            for ik,K in enumerate(Ks):
                for jn,N in enumerate(Ns):
                    print(dataparam)
                    noise_levels,num_samples = dataparam.get('noise_levels',[0]),dataparam.get('num_samples',[self.T])
                    param_path = f'{self.data_parameters_titles[a]}/N_{N}_K_{K}'
                    if self.exists_setup:
                        self.get_data_from_folder(param_path)
                    else:
                        self.create_data(dataparam,K,N)  
                    for p,num_sample in enumerate(num_samples):
                        for q,noise_level in enumerate(noise_levels):
                            if len(num_samples)*len(noise_levels) > 1:
                                param_path_extended = f'{param_path}/num_sample={num_sample}_noise_level={noise_level}'
                            if not os.path.exists(f'self.output_folder/res/{param_path_extended}') or len(os.listdir(f'self.output_folder/res/{param_path}'))==0:
                                for exp in range(self.N_exp):
                                    for algo in self.algos:
                                        algo.fill_experiment(self.setup['Rxs'][exp,p,q,:,:,:,:],self.setup['As'][exp,p,:,:,:],exp,self.setup['Winits'][exp,:,:,:],self.setup['Cinits'][exp,:,:,:],count_updates=self.updates)
                                        print(f'a = {a}, K = {K}, N = {N}, {num_sample} samples and noise = {noise_level} with algo {algo.name} :',algo.results['final_jisi'][exp],algo.results['total_times'][exp])
                                        if self.updates:
                                            print(f'Number of updates:', algo.results['number_updates'][exp]) 
                                self.store_in_folder(param_path_extended,'res')
                                

    def create_data(self,dataparam,K,N):
        epsilon,rho_bounds,lambda_,rank,noise_levels,num_samples = dataparam.get('epsilon',1),dataparam.get('rho_bounds',[0.4,0.6]),dataparam.get('lambda',0.1),dataparam.get('rank',K+10),dataparam.get('noise_levels',[0]),dataparam.get('num_samples',[self.T])
        self.setup['Rxs'] = np.zeros((self.N_exp,len(num_samples),len(noise_levels),K,K,N,N))
        self.setup['As'] = np.zeros((self.N_exp,len(num_samples),N,N,K))
        self.setup['Winits'] = np.zeros((self.N_exp,N,N,K))
        self.setup['Cinits'] = np.zeros((self.N_exp,K,K,N))
        for exp in range(self.N_exp):
            A = make_A(K,N)
            Sigma = make_Sigma(K,N,rank=rank,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
            S = make_S(Sigma,self.T)
            X = make_X(S,A)
            for p,num_sample in enumerate(num_samples):
                X_alt = X[:,:num_sample,:]
                X_,U = whiten_data_numpy(X_alt)
                A_ = np.einsum('nNk,Nvk->nvk',U,A)    
                Rx_ = np.einsum('NTK,MTJ->KJNM',X_,X_)/num_sample
                for q,noise_level in enumerate(noise_levels):
                    for k in range(K):
                        Rx_[k,k,:,:] += noise_level*np.eye(N)
                    self.setup['Rxs'][exp,p,q,:,:,:,:] = Rx_
                    self.setup['As'][exp,p,:,:,:] = A_
            self.setup['Winits'][exp,:,:,:] = make_A(K,N)
            self.setup['Cinits'][exp,:,:,:] = make_Sigma(K,N,rank=K+10)
            
            
    def compute_empirical_convergence(self,a,K,N,res_vars=['jisi','costs','times'],detailed=True,exp=0):
        track_params = {}
        for var in res_vars:
            track_params['track_' + var] = var in res_vars
            if detailed and not 'detailed' in var:
                res_vars.append('detailed_' + var)
        if 'diffs' in res_vars:
            res_vars += ['diffs_W','diffs_C']
        if self.exists_setup:
            self.get_setup_from_folder(a,K,N)
        else:
            self.create_setup(self.data_parameters[a],K,N)
        for algo in self.algos:
            res = algo.solve(self.setup['Datasets_cov'][exp,:,:,:,:],Winit=self.setup['Winits'][exp,:,:,:],Cinit=self.setup['Cinits'][exp,:,:,:],A=self.setup['Mixings'][exp,:,:,:],**track_params)
            for res_var in res_vars:
                if res_var in res.keys():
                    algo.results[res_var] = res[res_var]
        self.store_in_folder(a,K,N)

    def draw_empirical_convergence(self,a,K,N,res_type='costs',mode='time',algos=None):
        if algos == None:
            algos = self.algos
        # xlabels = {'time':'Time (s)','iter':'Nb of iterations'}
        # ylabels = {'jisi':'jISI score','costs':'cost function','diffs':'criteria'}
        output_path_individual = self.output_folder+ f'/{self.data_parameters_titles[a]}/N_{N}_K_{K}'
        fig,ax = plt.subplots()
        ax.set_yscale('log')
        for algo in algos:
            algo.fill_from_folder(output_path_individual)
            values = algo.results[res_type]
            if mode == 'time':
                if 'detailed' in res_type:
                    times = algo.results['detailed_times']
                else:
                    times = algo.results['times']              
                ax.plot(times,values,color=algo.color,label=algo.legend,linewidth=1)
            else:
                ax.plot(values,color=algo.color,label=algo.legend,linewidth=1)
            ax.legend(loc=1,fontsize=self.legend_fontsize)
        # ax.set_xlabel(xlabels[mode],fontsize=20)
        # ax.set_ylabel(ylabels[res_type],fontsize=20)
        ax.set_title('Empirical convergence',fontsize=self.title_fontsize)
        for extension in ['eps','png']:
            fig_path = os.path.join(output_path_individual, res_type + '_' + mode)
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            fig.savefig(fig_path,dpi=200,format=extension)
            plt.show()



# A corriger 
    # def make_charts(self,full=False):
    #     Ks,Ns = self.common_parameters
    #     for a,dataparam in enumerate(self.data_parameters):
    #         for ik,K in enumerate(Ks):
    #             for jn,N in enumerate(Ns):
    #                 os.makedirs(self.output_folder+f'/charts/{self.data_parameters_titles[a]}/N_{N}_K_{K}')
    #                 fig,ax = plt.subplots()
    #                 ax.set_xlabel('Time (s.)',fontsize=self.title_fontsize,labelpad=0)
    #                 ax.set_ylabel('jISI score',fontsize=self.title_fontsize,labelpad=0)
    #                 for algo in self.algos:
    #                     ax.errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),color=algo.color,label=algo.legend,elinewidth=2.5)
    #                 ax.set_yscale('log')
    #                 ax.grid(which='both')
    #                 # yticks = ax.get_yticks(minor=True)
    #                 # print(yticks)
    #                 # yticklabels = ['{:.0e}'.format(tick) for tick in yticks]
    #                 # ax.set_yticklabels(yticklabels)
    #                 # xticks = ax.get_xticks()
    #                 # xticklabels = ['{:.0e}'.format(tick) for tick in xticks]
    #                 # ax.set_xticklabels(xticklabels)
    #                 if self.legend:
    #                     fig.legend(loc=2,fontsize=self.legend_fontsize)
    #                 filename = 'comparison {} N = {} K = {}'.format(self.data_parameters_titles[a],N,K)
    #                 output_path = self.output_folder+f'/charts/{self.data_parameters_titles[a]}/N_{N}_K_{K}'
    #                 fig.savefig(output_path,dpi=200,bbox_inches='tight')
    #                 plt.close(fig)
    #         if full:
    #             fig,ax = plt.subplots(len(Ns),len(Ks),figsize=(12, 8))
    #             fig.text(0.5, 0.04, '$T$ (s.)', ha='center', fontsize=self.title_fontsize)
    #             fig.text(0.04, 0.5, '$jISI$ score', va='center', rotation='vertical', fontsize=self.title_fontsize)
    #             plt.yscale('log')
    #             for ik,K in enumerate(Ks):
    #                 for jn,N in enumerate(Ns):
    #                     if ik == 0:
    #                         ax[jn,ik].set_title('N = {}'.format(N))
    #                     for algo in self.algos:
    #                         ax[jn,ik].errorbar(np.mean(algo.times[a,ik,jn,:]),np.mean(algo.results[a,ik,jn,:]),
    #                                             yerr=np.std(algo.results[a,ik,jn,:]),xerr=np.std(algo.times[a,ik,jn,:]),
    #                                             color=algo.color,label=algo.legend,elinewidth=2.5)
    #             if self.legend:
    #                 fig.legend(loc=2,fontsize=self.legend_fontsize)
    #             filename = 'comparison {}.png'.format(self.data_parameters_titles[a])
    #             output_path = os.path.join(self.output_folder+'/charts/{}'.format(self.data_parameters_titles[a]), filename)
    #             fig.savefig(output_path,dpi=200,bbox_inches='tight')
    #             plt.close(fig)




                    

        

