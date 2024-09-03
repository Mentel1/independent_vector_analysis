import torch
import torch.optim as optim
from model import *
from data import *
from tools import *
from functions import *
from torch.utils.data import DataLoader
from torch import nn
import os
import sys










""" 
for name, param in model.alpha_network.named_parameters():
    print(name, param.requires_grad)

def print_gradients(module, grad_input, grad_output):
    print(f'Gradients for {module}: {grad_output}')




 """





# After training, you can access the trained model parameters, including the optimized alpha values, using model.parameters().


## Model parameters

gamma_c = 1
gamma_w = 0.99
eps = 1e-12
nu = 0.5
zeta = 1e-3 


# Hyperparameters

T = 10000
K = 3
N = 2

lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']


input_dim = N * N * K + K * K * N
N_updates_W = 10
N_updates_C = 1
dataset_size = 100
learning_rate = 1e-2
num_epochs = 1
batch_size = 4
num_layers = 300

folders = ['kjerkfj','kjsvkj','kndskjn']



class U_TITAN(nn.Module):
    """
    Includes the main training and testing methods of U_TITAN.

    """

    def __init__(self, folders = folders, mode='end-to-end', T=T, K=K, N=N, metaparameters_multiparam=metaparameters_multiparam, size = dataset_size, input_dim = input_dim,
                    lr = learning_rate, N_updates_W = N_updates_W,N_updates_C = N_updates_C ,num_epochs = num_epochs, batch_size = batch_size, num_layers=num_layers, gamma_c=gamma_c, gamma_w=gamma_w, eps=eps, nu=nu, zeta=zeta):
            """
            Parameters
            ----------
                test_conditions    (list): list of 5 elements, the name of the blur kernel (str), the noise level (str), the range of the noise standard deviation (list), 
                                        the image size (numpy array), minimal and maximal pixel values (list)
                folders            (list): list of str, paths to the folder containing (i) the test sets, (ii) the training, (iii) saved models
                mode                (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                                        'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model (default is 'first_layer')
                lr_first_layer     (list): list of two elements, first one is the initial learning rate to train the first layer, 
                                        second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-2,5])    
                lr_greedy          (list): list of two elements, first one is the initial learning rate to train the layers, 
                                        second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [5e-3,5])
                lr_last_layers_lpp (list): list of two elements, first one is the initial learning rate to train the last 10 layers conjointly with lpp, 
                                        second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-3,50])  
                nb_epochs          (list): list of three integers, number of epochs for training the first layer, the remaining layers, 
                                        and the last 10 layers + lpp, respectively (default is [40,40,600])      
                nb_blocks           (int): number of unfolded iterations (default is 40)    
                nb_greedy_blocks    (int): number of blocks trained in a greedy way (default is 30)
                batch_size         (list): list of three integers, number of images per batch for training, validation and testing, respectively (default is [10,10,1])                
                loss_type           (str): name of the training loss (default is 'SSIM')  
            """
            super(U_TITAN, self).__init__() # On peut sans doute enlever les commandes entre parenthèses dans le 'super'     
            # unpack information about test conditions and saving folders
            self.T = T
            self.K = K
            self.N = N
            self.input_dim = input_dim
            self.metaparameters_multiparam = metaparameters_multiparam
            self.dataset_size = size
            self.path_test, self.path_train, self.path_save = folders
            self.mode  = mode #'first_layer' or 'greedy' or 'last_layers_lpp' or 'test'
            # training information
            self.lr = lr
            self.num_epochs = num_epochs 
            self.num_layers = num_layers
            self.batch_size = batch_size 
            self.dtype = torch.cuda.FloatTensor
            self.model = myModel(input_dim, N_updates_W, N_updates_C, num_layers=num_layers, gamma_c=gamma_c, gamma_w=gamma_w, eps=eps, nu=nu, zeta=zeta,B=batch_size).cuda()
            self.loss_fun = ISI_loss()



    def CreateLoader(self, dataset, batch_size):
        """
        Create a DataLoader object from a dataset

        Parameters
        ----------
            dataset (torch.utils.data.Dataset): dataset to be loaded
            batch_size (int): number of images per batch

        Returns
        -------
            DataLoader object
        """

        train_data = MyDataset(self.T, self.K, self.N, self.metaparameters_multiparam, size = self.dataset_size)
        #val_data   = MyDataset(self.T, self.K, self.N, self.metaparameters_multiparam, size = self.dataset_size)
        self.size_train = train_data.size
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        #self.val_loader   = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)


    def CreateFolders(self,layer):
        
        """
        
            
            Create folders to save the results of the training and testing

            

            if self.mode=='first_layer' or self.mode=='greedy':
                name = 'block_'+str(layer)
                if not os.path.exists(os.path.join(self.path_save,name)):
                    os.makedirs(os.path.join(self.path_save,name,'training'))


            # create the folder to save the results of the training
            if not os.path.exists(self.path_save):
                os.makedirs(self.path_save)
            # create the folder to save the results of the testing
            if not os.path.exists(self.path_test):
                os.makedirs(self.path_test)
            # create the folder to save the results of the testing
            if not os.path.exists(self.path_test):
                os.makedirs(self.path_test)
        
        """


    def train(self,layer=0):
        """
        Trains U_TITAN.
        Parameters
        ----------
            layer (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """   
        if self.mode == 'end-to-end':

            # trains the whole network
            print('=================== End-to-end training ===================')
            # to store results
            loss_epochs       =  np.zeros(self.num_epochs)
            isi_train   =  np.zeros(self.num_epochs)
            isi_val     =  np.zeros(self.num_epochs)
            loss_min_val      =  float('Inf')
            #self.CreateFolders(layer)
            #folder = os.path.join(self.path_save,'Layer_'+str(layer))
            dataset = MyDataset(T, K, N, metaparameters_multiparam, size = dataset_size)
            self.CreateLoader(dataset=dataset, batch_size=self.batch_size) 
            # defines the optimizer
            optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.num_epochs):
                
                self.model.train()
                epoch_loss = 0
                # goes through all minibatches

                for i,minibatch in enumerate(self.train_loader):
                    [X, A,Winit,Cinit] = minibatch
                    #print("Winit size: ", Winit.size())
                    
                    W_predicted,_ = self.model(X,A,Winit,Cinit,self.mode,layer)
                    #print("w predicted requires grad: ", W_predicted.requires_grad)

                    # Computes and prints loss
                    loss = self.loss_fun(W_predicted, A)
                    #print("loss requires grad: ", loss.requires_grad)
                    epoch_loss += torch.Tensor.item(loss)
                    sys.stdout.write('\r Epoch %d/%d, minibatch %d/%d, loss: %.4f \n' % (epoch+1,self.num_epochs,i+1,self.size_train//self.batch_size,loss))

                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    # Check gradients
                    """for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(f"Parameter '{name}' has no gradient")
                        else:
                            pass
                            print(f"Parameter '{name}' gradient mean: {param.grad.mean().item()}")  """
                    loss.backward()
                    optimizer.step()


                loss_epochs[epoch] = epoch_loss / len(self.train_loader)
                print('Epoch %d/%d, loss: %.4f' % (epoch+1,self.num_epochs,loss_epochs[epoch]))

            # Get a list of parameters
            parameters = list(self.model.parameters())

            # Print the parameters
            for i, parameter in enumerate(parameters):
                print(f"Parameter {i}: {parameter}")
                    


        elif self.mode == 'greedy':
            # trains the next layer
            print('=================== Layer number %d ==================='%(layer))
            # to store results
            loss_epochs       =  np.zeros(self.num_epochs)
            isi_train   =  np.zeros(self.num_epochs)
            isi_val     =  np.zeros(self.num_epochs)
            loss_min_val      =  float('Inf')
            self.CreateFolders(layer)
            #folder = os.path.join(self.path_save,'Layer_'+str(layer))
            dataset = MyDataset(T, K, N, metaparameters_multiparam, size = dataset_size)
            self.CreateLoader(dataset=dataset, batch_size=self.batch_size)
            # puts first blocks in evaluation mode: gradient is not computed
            #self.model.GradFalse(layer,self.mode) 
            # defines the optimizer
            lr = self.lr
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.num_epochs):
                
                self.model.Layers[layer].train() # training mode
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader):
                    [X, A] = minibatch  # gets the minibatch
                    #print("minibatch size: ", minibatch[0].size())
                    X = X[0]
                    A = A[0]
                    print("X size: ", X.size())
                    Rx = cov_X(X)
                    print("Rx size: ", Rx.size())
                    W,C = initialize(N,K,X=X,Rx=Rx)

                    W_predicted,C_predicted = self.model(Rx,W,C,self.mode,layer)

                    # Computes and prints loss
                    loss = self.loss_fun(W_predicted, A)
                    loss_epochs[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r Epoch %d/%d, minibatch %d/%d, loss: %.4f \n' % (epoch+1,self.num_epochs,i+1,self.size_train//self.batch_size,loss))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Check gradients
                    """ for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(f"Parameter '{name}' has no gradient")
                        else:
                            print(f"Parameter '{name}' gradient mean: {param.grad.mean().item()}") """


            # tests on validation set
                
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Layer ' + str(layer) + ' is done.')
            print('-----------------------------------------------------------------')
            
            # calls the same function to start training of next block 
            if layer < self.num_layers-1:
                self.train(layer=layer+1)
            else:
                print('Training of all layers is done.')
        
        ###############################################################################################################
        






        
test = U_TITAN()
test.train()
