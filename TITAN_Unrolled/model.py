"""
UTitan model classes
Classes
-------
    ISI_loss  : defines the ISI training loss 
    nn_alpha    : predicts the regularisation parameter
    W_iter     : computes the updates of W
    C_iter    : computes the updates of C
    Block      : one layer in U_TITAN
    myModel    : U_TITAN model


@author: Gaspard Blaise
@date: 11/06/2024
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functions import *
from tools import *
from data import *
from torch.utils.data import DataLoader
#from dataloader_file import *



class MyDataset(Dataset):
    def __init__(self, T, K, N, metaparameters_multiparam,size):
        self.T = T
        self.K = K
        self.N = N
        self.metaparameters_multiparam = metaparameters_multiparam
        self.size = size
        self.half_size = size // 2

    def __len__(self):
        # retourne la taille du dataset
        return self.size  # remplacez par la taille réelle de votre dataset

    def __getitem__(self, idx):
        # Generates a new sample from the dataset
        if idx < self.half_size:
            rho_bounds, lambda_ = self.metaparameters_multiparam[1]  # Use case 2
        else:
            rho_bounds, lambda_ = self.metaparameters_multiparam[3]  # Use case 4
        #print(f"rho_bounds: {rho_bounds}, lambda_: {lambda_}")
        X, A = generate_whitened_problem(self.T, self.K, self.N, epsilon=1, rho_bounds=rho_bounds, lambda_=lambda_)
        Winit, Cinit = initialize(self.N, self.K)
        return X, A, Winit, Cinit




class ISI_loss():
    """
    Defines the ISI training loss.
    Attributes
    ----------
        ISI : function computing the ISI Score 
    """
    def __init__(self): 
        super().__init__()
        
    def __call__(self, input, target):
        """
        Computes the training loss.
        Parameters
        ----------
            input  (torch.FloatTensor): restored images, size n*c*h*w 
            target (torch.FloatTensor): ground-truth images, size n*c*h*w
        Returns
        -------
            (torch.FloatTensor): mean ISI Score of the batch, size 1 
        """
        batch_size = input.shape[0]
        isi_scores = []

        for i in range(batch_size):
            W = input[i] # Select the i-th element in the batch and add a batch dimension
            A = target[i]  # Select the i-th element in the batch and add a batch dimension
            score = joint_isi(W, A)
            isi_scores.append(score)

        isi_scores = torch.stack(isi_scores)  # Stack scores into a tensor
        return torch.mean(isi_scores)



class FCNN_alpha(nn.Module):
    """
    Predicts the regularization parameter alpha given W and C.
    Attributes
    ----------
        fc1 (torch.nn.Linear): fully connected layer
        fc2 (torch.nn.Linear): fully connected layer
        soft (torch.nn.Softplus): Softplus activation function
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softplus()
        
    def forward(self, W, C):
        """
        Computes the regularization parameter alpha.
        Parameters
        ----------
            W (torch.FloatTensor): input tensor W, size [batch_size, W_size]
            C (torch.FloatTensor): input tensor C, size [batch_size, C_size]
        Returns
        -------
            torch.FloatTensor: alpha parameter, size [batch_size, 1]
        """
        B, K, _, N = C.shape
        # Flatten W to size [N*N*K]
        W = W.reshape(W.size(0), -1)
        # Flatten C to size [K*K*N]
        C = C.reshape(W.size(0), -1)
        # Concatenate W and C along the last dimension
        x = torch.cat((W, C), dim=-1)
        x = self.soft(self.fc1(x))
        x = self.fc2(x)
        x = self.soft(x)
        x = x.squeeze(dim=-1)
        return x






class W_iter(nn.Module):

    def __init__(self,N_updates_W,learning_mode):
        super(W_iter, self).__init__()
        self.N_updates_W = N_updates_W
        self.mode = learning_mode

    def inertial_step(self,beta_w,W,W_old):
        #print("W shape",W.shape)
        #print("beta_w shape",beta_w.shape)
        beta_w = beta_w.view(-1, 1, 1, 1)
        W = W + beta_w * (W - W_old)
        return W
    
    def gradient_step(self,Rx,c_w,W,C):
        c_w = c_w.view(-1, 1, 1, 1)
        W = W - c_w * grad_H_W(W, C, Rx)
        return W

    def prox_step(self,c_w,W):
        return prox_f(W, c_w)

    def update(self,Rx,W,W_j_1,C,c_w,beta_w):
        if self.mode == 'with_inertial':
            W_inertial = self.inertial_step(beta_w,W,W_j_1)
            W_gradient = self.gradient_step(Rx,c_w,W_inertial,C)
            W_prox = self.prox_step(c_w,W_gradient)
            return W_prox,W
        
        if self.mode == 'no_inertial':
            W_gradient = self.gradient_step(Rx,c_w,W,C)
            W_prox = self.prox_step(c_w,W_gradient)
            return W_prox,W
        else :  
            W_inertial = self.inertial_step(beta_w,W,W_j_1)
            W_gradient = self.gradient_step(Rx,c_w,W_inertial,C)
            W_prox = self.prox_step(c_w,W_gradient)
            return W_prox,W
    
    def forward(self,Rx,W,W_j_1,C,c_w,beta_w):
        for _ in range(self.N_updates_W):
            W,W_j_1 = self.update(Rx,W,W_j_1,C,c_w,beta_w)
        return W,W_j_1
    


class C_iter(nn.Module):
    def __init__(self,N_updates_C,learning_mode):
        super(C_iter, self).__init__()
        self.N_updates_C = N_updates_C
        self.mode = learning_mode

    def inertial_step(self,beta_c,C,C_old):
        beta_c = beta_c.view(-1, 1, 1, 1)
        C = C + beta_c * (C - C_old)
        return C
    
    def gradient_step(self,Rx,c_c,C,W,alpha):
        c_c = c_c.view(-1, 1, 1, 1)
        C = C - c_c * grad_H_C_reg(W, C, Rx, alpha)
        return C

    def prox_step(self,c_c,C,eps):
        return prox_g(C, c_c, eps)
    
    def update(self,Rx,C,C_j_1,W_updated,c_c,beta_c,alpha,eps):
        if self.mode == 'with_inertial':
            C_inertial = self.inertial_step(beta_c,C,C_j_1)
            C_gradient = self.gradient_step(Rx,c_c,C_inertial,W_updated,alpha)
            C_prox = self.prox_step(c_c,C_gradient,eps)
            return C_prox,C
        
        if self.mode == 'no_inertial':
            C_gradient = self.gradient_step(Rx,c_c,C,W_updated,alpha)
            C_prox = self.prox_step(c_c,C_gradient,eps)
            return C_prox,C
        else :
            C_inertial = self.inertial_step(beta_c,C,C_j_1)
            C_gradient = self.gradient_step(Rx,c_c,C_inertial,W_updated,alpha)
            C_prox = self.prox_step(c_c,C_gradient,eps)
            return C_prox,C

    def forward(self,Rx,C,C_j_1,W_updated,c_c,beta_c,alpha,eps):
        C_i_1 = C.clone()
        for _ in range(self.N_updates_C):
            C,C_j_1 = self.update(Rx,C,C_j_1,W_updated,c_c,beta_c,alpha,eps)
        return C,C_j_1,C_i_1




class Block(nn.Module):

    """
    One layer in U_TITAN.
    Attributes
    ----------
        nn_bar                           (Cnn_bar): computes the barrier parameter
        soft                    (torch.nn.Softplus): Softplus activation function
        gamma                (torch.nn.FloatTensor): stepsize, size 1 
        reg_mul,reg_constant (torch.nn.FloatTensor): parameters for estimating the regularization parameter, size 1
        delta                               (float): total variation smoothing parameter
        IPIter                             (IPIter): computes the next proximal interior point iterate
    """


    def __init__(self,K,N,input_dim, N_updates_W,N_updates_C,gamma_c,gamma_w,eps,nu,zeta,learning_mode):
    
        super().__init__()
        #self.NN_alpha = FCNN_alpha(input_dim, 32)
        #self.reg_mul      = nn.Parameter(torch.FloatTensor([-7]).cuda()) 
        #self.reg_constant = nn.Parameter(torch.FloatTensor([-5]).cuda())
        self.W_iter = W_iter(N_updates_W,learning_mode)
        self.C_iter = C_iter(N_updates_C,learning_mode)

        #self.alpha = nn.Parameter(torch.FloatTensor([1]).cuda())
        self.alpha = nn.Parameter(torch.empty(1).cuda())

        torch.nn.init.normal_(self.alpha, mean=0, std=0.01)

        # Manually apply He initialization for a scalar
        #std = math.sqrt(2.)  # Typical scale factor for He initialization
        #with torch.no_grad():
            #self.alpha.normal_(0, std)

        if learning_mode == 'with_inertial':
            self.gamma_w = nn.Parameter(torch.FloatTensor([0.5]).cuda())
            self.gamma_c = nn.Parameter(torch.FloatTensor([0.5]).cuda())
            self.beta_w = nn.Parameter(torch.empty(1).cuda())
            self.beta_c = nn.Parameter(torch.empty(1).cuda())

            torch.nn.init.normal_(self.beta_w, mean=0, std=0.01)
            torch.nn.init.normal_(self.beta_c, mean=0, std=0.01)
            #torch.nn.init.uniform_(self.beta_w, a=-1, b=1)
            #print("beta_w",self.beta_w)
            #torch.nn.init.uniform_(self.beta_c, a=-1, b=1) 
            #print("beta_c",self.beta_c)

        elif learning_mode == 'no_inertial':
            self.gamma_w = nn.Parameter(torch.FloatTensor([1]).cuda())
            self.gamma_c = nn.Parameter(torch.FloatTensor([1]).cuda())

        else:
            self.gamma_w = gamma_w
            self.gamma_c = gamma_c
         
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps
        self.nu = nu
        self.zeta = zeta
        self.learning_mode = learning_mode
    
    def get_coefficients(self,rho_Rx,alpha,C,C_i_1):

        #assert alpha.requires_grad, "alpha does not require grad"

        B, K, _, N = C.shape
        #print("alpha",alpha.shape)
        #print("rho_Rx",rho_Rx.shape)
        
        part1 = (self.gamma_w*alpha)/(1-self.gamma_w)
        part2 = rho_Rx*2*K*(1+torch.sqrt(2/(alpha*self.gamma_c)))


        l_sup = torch.max(part1,part2)
        #print("l_sup",l_sup)

        expr3 = torch.tensor(self.gamma_c**2 / K**2).expand(B).cuda()
        expr4 = alpha * self.gamma_w / ((1 + self.zeta) * (1 - self.gamma_w) * l_sup)
        expr5 = rho_Rx / ((1 + self.zeta) * l_sup)

        #print("expr3",expr3)
        #print("expr4",expr4)
        #print("expr5",expr5)

        # Compute the minimum of the three expressions
        C0 = torch.min(torch.min(expr3, expr4), expr5)
        #print("C0",C0)

        
        L_inf = (1+self.zeta)*C0*l_sup
        #print("L_inf",L_inf)
        L_w_prev = torch.max(L_inf,lipschitz(C_i_1,rho_Rx))
        #print("C_i_1",C_i_1)
        #print("lip",lipschitz(C_i_1,rho_Rx))
        L_w = torch.max(L_inf,lipschitz(C,rho_Rx))
        #print("L_w",L_w)

        c_w = self.gamma_w/L_w
        #print("c_w",c_w)
        beta_w = (1-self.gamma_w)*torch.sqrt(C0*self.nu*(1-self.nu)*L_w_prev/L_w)
        #print("beta_w",beta_w)
        
        c_c = self.gamma_c/alpha
        beta_c = torch.sqrt(C0 * self.nu*(1-self.nu))

        return c_w,beta_w,c_c,beta_c



    def forward(self,Rx,rho_Rx,W,W_j_1,C,C_j_1,C_i_1):
        """
        Computes the next iterate, output of the layer.
        Parameters
        ----------
      	    x            (torch.nn.FloatTensor): previous iterate, size n*c*h*w
            Ht_x_blurred (torch.nn.FloatTensor): Ht*degraded image, size n*c*h*w
            std_approx   (torch.nn.FloatTensor): approximate noise standard deviation, size n*1
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterate, output of the layer, n*c*h*w
        """

        alpha = self.softplus(self.alpha)
        #print("alpha",alpha)

    
        #print("alpha shape",alpha.shape)
        #alpha = self.NN_alpha(W,C)
        #alpha = torch.tensor(1.0,requires_grad=True)

        if self.learning_mode == 'with_inertial':
            gamma_w = 0.3 + 2 * (self.tanh(self.gamma_w) + 1)  # gamma_w in [0.3, 10.3]
            gamma_c = 0.3 + 2 * (self.tanh(self.gamma_c) + 1)  # gamma_c in [0.3, 10.3]
            c_w = gamma_w / lipschitz(C, rho_Rx)
            c_c = gamma_c / alpha
            beta_w = self.softplus(self.beta_w)
            beta_c = self.softplus(self.beta_c)

        elif self.learning_mode == 'no_inertial':
            gamma_w = 0.3 + 5 * (self.tanh(self.gamma_w) + 1)  # gamma_w in [0.3, 10.3]
            gamma_c = 0.3 + 5 * (self.tanh(self.gamma_c) + 1)  # gamma_c in [0.3, 10.3]
            c_w = gamma_w / lipschitz(C,rho_Rx)
            c_c = gamma_c / alpha
            beta_w = 0
            beta_c = 0
            
        else:
            c_w, beta_w, c_c ,beta_c = self.get_coefficients(rho_Rx, alpha, C, C_i_1)
        
        W, W_j_1  = self.W_iter(Rx, W, W_j_1, C, c_w, beta_w)
        C, C_j_1, C_i_1  = self.C_iter(Rx, C, C_j_1, W, c_c, beta_c, alpha, self.eps)


        #W_updated,W_j_1 = self.W_iter(Rx,W,W_j_1,C,c_w,beta_w)
        #C_updated,C_j_1,C_i_1= self.C_iter(Rx,C,C_j_1,W_updated,c_c,beta_c,alpha,self.eps)
        return W, W_j_1, C,  C_j_1, C_i_1




class myModel(nn.Module):
    """
    U_TITAN model.
    Attributes
    ----------
        blocks (list): list of Blocks
    """
    def __init__(self,K,N,input_dim, N_updates_W,N_updates_C,num_layers,gamma_c,gamma_w,eps,nu,zeta,learning_mode):
        super().__init__()
        self.Layers = nn.ModuleList([Block(K,N,input_dim,N_updates_W,N_updates_C,gamma_c,gamma_w,eps,nu,zeta,learning_mode) for _ in range(num_layers)])
        self.isi_scores = []



    def forward(self,X,A,mode,layer=0):
        """
        Computes the next iterate, output of the model.
        Parameters
        ----------
      	    W,C            (torch.nn.FloatTensor): previous iterates, size B*N*N*K, B*K*K*N
            Rx             (torch.nn.FloatTensor): covariance matrix of the input, size K*K*N
            mode          (str): 'first_layer' if training the first layer, 'greedy' if training one layer at a time, 'test' if testing
            layer                         (int): layer-1 is the layer to be trained (default is 0)
            save_gamma_mu_lambda          (str): indicates if the user wants to save the values of the estimated hyperparameters, 
                                                 path to the folder to save the hyperparameters values or 'no' 
        Returns
        -------
       	    (torch.FloatTensor): next iterates, output of the model, n*N*N*K, n*K*K*N
        """
        B,N,_,K = X.shape
        #print("X shape",X.shape)
        Rx = cov_X(X)
        rho_Rx = spectral_norm_extracted(Rx,K,N)
        #print("Rx shape",Rx.shape)
        W,C = initialize(N,K,B)
        


        if mode=='first_layer' or mode=='greedy':
            C_old = C.clone()
            W,C,C_old = self.Layers[layer](rho_Rx,W,C,C_old)

        elif mode=='end-to-end':
            W_j_1 = W.clone()
            C_i_1 = C.clone()
            C_j_1 = C.clone()
            for i in range(len(self.Layers)):
                #alpha = self.soft(self.alphas[i])
                #assert self.Layers[i].alpha.requires_grad, f"alpha {i} does not require grad"
                W,W_j_1,C,C_j_1,C_i_1 = self.Layers[i](Rx,rho_Rx,W,W_j_1,C,C_j_1,C_i_1)
                #print("W",W)
                #print("ISI Score Layer ",i+1,": ",ISI_loss()(W, A))
        
        elif mode=='test':
            W_j_1 = W.clone()
            C_i_1 = C.clone()
            C_j_1 = C.clone()
            for i in range(len(self.Layers)):
                W,W_j_1,C,C_j_1,C_i_1 = self.Layers[i](Rx,rho_Rx,W.detach(),W_j_1.detach(),C.detach(),C_j_1.detach(),C_i_1.detach())
                #print("ISI Score Layer ",i+1,": ",joint_isi(W, A))
                self.isi_scores.append(ISI_loss()(W, A))

        return W,C         
    



















"""
class DeterministicBlock(nn.Module):
    def __init__(self, Rx, rho_Rx, gamma_c, gamma_w, eps, nu, zeta):
        super(DeterministicBlock, self).__init__()
        self.Rx = Rx
        self.rho_Rx = rho_Rx
        self.gamma_c = gamma_c
        self.gamma_w = gamma_w
        self.eps = eps 
        self.nu = nu
        self.zeta = zeta
        self.device = 'cuda:0'    

    def forward(self, W, C, alpha, L_w_prev):
        K = W.shape[2]
        alpha = alpha.to(self.device)
        #print(alpha.device)

        l_sup = max((self.gamma_w*alpha)/(1-self.gamma_w),self.rho_Rx*2*K*(1+torch.sqrt(2/(alpha*self.gamma_c))))
        C0 = min(self.gamma_c**2/K**2,(alpha*self.gamma_w/((1+self.zeta)*(1 - self.gamma_w)*l_sup)),(self.rho_Rx/((1+self.zeta)*l_sup)))
        l_inf = (1+self.zeta)*C0*l_sup

        c_c = self.gamma_c / alpha
        beta_c = torch.sqrt(C0*self.nu*(1-self.nu))
        L_w = max(l_inf,lipschitz(C,self.rho_Rx))
        c_w = self.gamma_w / L_w
        beta_w = (1 - self.gamma_w) * torch.sqrt(C0 * self.nu * (1 - self.nu) * L_w_prev / L_w)
        W_prev = W.clone()
        W_prev = W_prev.to(self.device)
        #print(W.device)	

        for _ in range(10):
            
            W_tilde = W + beta_w * (W - W_prev)
            grad_W = grad_H_W(W_tilde, C, self.Rx)
            W_bar = W_tilde - c_w * grad_W
            W_prev = W.clone()
            W = prox_f(W_bar, c_w)

        C_prev = C.clone()
        beta_c = torch.sqrt(C0 * self.nu * (1 - self.nu))
        C_tilde = C + beta_c * (C - C_prev)
        grad_C = grad_H_C_reg(W, C_tilde, self.Rx, alpha)
        C_bar = C_tilde - c_c * grad_C
        C_prev = C.clone()
        C = prox_g(C_bar, c_c, self.eps)

        return W, C, L_w




# Define the TitanLayer
class TitanLayer(nn.Module):
    def __init__(self, Rx, rho_Rx, gamma_c, gamma_w, eps, nu, input_dim, zeta):
        super(TitanLayer, self).__init__()
        self.alpha_net = AlphaNetwork(input_dim)
        self.deterministic_block = DeterministicBlock(Rx, rho_Rx, gamma_c, gamma_w, eps, nu, zeta)

    def forward(self, W, C, L_w_prev):
        alpha = self.alpha_net(W, C)
        W, C, L_w = self.deterministic_block(W, C, alpha, L_w_prev)
        return W, C, L_w, alpha
    




class TitanIVAGNet(nn.Module):
    def __init__(self, input_dim, num_layers=20, gamma_c=1, gamma_w=0.99, eps=1e-12, nu=0.5, zeta=0.1):
        super(TitanIVAGNet, self).__init__()
        self.num_layers = num_layers
        self.gamma_c = torch.tensor(gamma_c)
        self.gamma_w = torch.tensor(gamma_w)
        self.eps = torch.tensor(eps)
        self.nu = torch.tensor(nu)  
        self.zeta = torch.tensor(zeta)
        self.alpha_network = AlphaNetwork(input_dim)
        self.alphas = [torch.FloatTensor([1]).to('cuda') for _ in range(num_layers)]
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            TitanLayer(None, None, gamma_c, gamma_w, eps, nu, input_dim, zeta)
            for _ in range(num_layers)
        ])
    


    def initialize_L_w(self, C, rho_Rx, K):
        l_sup = max((self.gamma_w * self.alphas[0]) / (1 - self.gamma_w), rho_Rx * 2 * K * (1 + torch.sqrt(2 / (self.alphas[0] * self.gamma_c))))
        C0 = min(self.gamma_c**2 / K**2, (self.alphas[0] * self.gamma_w / ((1 + self.zeta) * (1 - self.gamma_w) * l_sup)), (rho_Rx / ((1 + self.zeta) * l_sup)))
        l_inf = (1 + self.zeta) * C0 * l_sup
        return max(l_inf, lipschitz(C, rho_Rx))
    
    
    def forward(self, X, A):
        N,_,K = X.shape
        input_dim = N * N * K + K * K * N
        Rx = cov_X(X)
        rho_Rx = spectral_norm_extracted(Rx, K, N)

        
        W, C = initialize(N, K, init_method='random', Winit=None, Cinit=None, X=X, Rx=Rx, seed=None)
        
        L_w_prev = self.initialize_L_w(C, rho_Rx, K)


        for i, layer in enumerate(self.layers):

            layer.deterministic_block = DeterministicBlock(Rx, rho_Rx, self.gamma_c, self.gamma_w, self.eps, self.nu, self.zeta)  # Ensure each layer has its own deterministic block
            W, C, L_w, alpha = layer(W, C, L_w_prev)
            L_w_prev = L_w
            self.alphas[i] = alpha
            

        
        isi_score = joint_isi(W, A)

        return W, C, isi_score

"""