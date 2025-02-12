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
from tqdm import tqdm
import time
import gc
















# After training, you can access the trained model parameters, including the optimized alpha values, using model.parameters().




class U_TITAN(nn.Module):
    """
    Includes the main training and testing methods of U_TITAN.

    """

    def __init__(self, folders, model_name, mode, T, K, N, input_dim, lr, N_updates_W, N_updates_C, num_epochs, batch_size, dataset_size, num_layers, gamma_c, gamma_w, eps, nu, zeta, learning_mode):
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
            super().__init__()           
            # unpack information about test conditions and saving folders
            self.T = T
            self.K = K
            self.N = N
            self.input_dim = input_dim
            self.model_name = model_name
            self.size = dataset_size
            self.path_test, self.path_train, self.path_val, self.path_save = folders
            self.mode  = mode #'first_layer' or 'greedy' or 'last_layers_lpp' or 'test'
            # training information
            self.lr = lr
            self.num_epochs = num_epochs 
            self.num_layers = num_layers
            self.batch_size = batch_size 
            self.dtype = torch.cuda.FloatTensor
            self.model = myModel(model_name,K,N,input_dim, N_updates_W, N_updates_C, num_layers=num_layers, gamma_c=gamma_c, gamma_w=gamma_w, eps=eps, nu=nu, zeta=zeta,learning_mode=learning_mode).cuda()
            self.loss_fun = ISI_loss()
            self.isi_train   =  np.zeros(self.num_epochs)
            self.isi_val     =  np.zeros(self.num_epochs)
            self.isi_test    =  []
            self.test_mean_time_per_data = 0
            self.test_mean_time_per_batch = 0



    def CreateLoader(self):
        """
        Create a DataLoader object from a dataset

        Returns
        -------
            DataLoader object
        """
        if self.mode == 'end-to-end':
            train_data  = MyDatasetFolder(self.path_train, self.K, self.N, self.size)
            val_data    = MyDatasetFolder(self.path_val, self.K, self.N, self.size)



        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.size_train   = len(train_data)
        self.size_val     = len(val_data)



    def CreateFolders(self,training_number):
        
        """
        Create folders to save the results of the training and testing
        """

        if self.mode == 'end-to-end':

            folder = os.path.join(self.path_save,'Training_'+str(training_number))

            if not os.path.exists(folder):  
                os.makedirs(folder)
                


    def train(self,layer=0):
        """
        Trains U_TITAN.
        Parameters
        ----------
            training_number (int): number of the training (default is 0)
            layer (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """   
        if self.mode == 'end-to-end':

            # trains the whole network
            print(f'=================== End-to-end training ===================')
            # to store results 

            loss_min_val = float('Inf')
            #self.CreateFolders(training_number)
            #folder = os.path.join(self.path_save,'Layer_'+str(layer))
            self.CreateLoader() 
            # defines the optimizer
            optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=1e-3)
            # Initialize learning rate scheduler
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)       
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.num_epochs):
                
                self.model.train()
                epoch_loss = 0
                # goes through all minibatches

                for i,minibatch in enumerate(self.train_loader):
                    #print("minibatch: ", minibatch)
                    X, A = minibatch  # gets the minibatch
                    #print("A : ", A)
                    #print("Winit size: ", Winit.size())

                    # Debugging: Check for NaNs in input data
                    if torch.isnan(X).any() or torch.isnan(A).any():
                        print(f"NaNs detected in input data at epoch {epoch}, batch {i+1}")
                        return
                    
                    W_predicted,_= self.model(X,A,self.mode)
                    #print("w predicted requires grad: ", W_predicted.requires_grad)

                    # Debugging: Check for NaNs in model output
                    if torch.isnan(W_predicted).any():
                        print(f"NaNs detected in model output at epoch {epoch}, batch {i+1}")
                        return

                    # Computes and prints loss
                    loss = self.loss_fun(W_predicted, A)

                    # Debugging: Check for NaNs in loss
                    if torch.isnan(loss).any():
                        print(f"NaNs detected in loss at epoch {epoch}, batch {i+1}")
                        return

                    #print("loss requires grad: ", loss.requires_grad)
                    epoch_loss += torch.Tensor.item(loss)
                    #sys.stdout.write('\r Epoch %d/%d, minibatch %d/%d, loss: %.4f \n' % (epoch+1,self.num_epochs,i+1,self.size_train//self.batch_size,loss))

                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient Clipping
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Debugging: Check for NaNs in gradients
                    """ for name, param in self.model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"NaNs detected in gradients of {name} at epoch {epoch}, batch {i+1}")
                            return

                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(f"Parameter '{name}' has no gradient")
                        else:
                            print(f"Parameter '{name}' gradient mean: {param.grad.mean().item()}") 
                            print(f"Parameter '{name}' value: {param.mean().item()}")  """

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    #print(f"batch {i} done")
                    #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")    
                



                self.isi_train[epoch] = epoch_loss / len(self.train_loader)
                print('Epoch %d/%d, training loss: %.4f' % (epoch+1,self.num_epochs,self.isi_train[epoch]))

                # Sauvegarder le modèle
                model_file = os.path.join(self.path_save, f'model_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), model_file)
                


                # tests on validation set
                self.model.eval()
                loss_current_val = 0
                with torch.no_grad():
                    for i,minibatch in enumerate(self.val_loader):
                        X, A = minibatch

                        if torch.isnan(X).any() or torch.isnan(A).any():
                            print(f"NaNs detected in input data at epoch {epoch}, batch {i}")
                            print(f"X: {X}")
                            print(f"A: {A}")
                            return
                    

                        W_predicted,_ = self.model(X,A,self.mode)
                        
                        if torch.isnan(W_predicted).any():
                            print(f"NaNs detected in model output at epoch {epoch}, batch {i}")
                            return
                        
                    
                        loss_current_val += torch.Tensor.item(self.loss_fun(W_predicted, A))
                self.isi_val[epoch] = loss_current_val / len(self.val_loader) 
                print('Epoch %d/%d, validation loss: %.4f' % (epoch+1,self.num_epochs,self.isi_val[epoch]))
                scheduler.step(self.isi_val[epoch])

                if loss_current_val < loss_min_val:
                    loss_min_val = loss_current_val
                    best_model_file = os.path.join(self.path_save, 'best_val.pth')
                    torch.save(self.model.state_dict(), best_model_file)



            # training is finished
            print('-----------------------------------------------------------------')
            print('Training is done.')
            print('-----------------------------------------------------------------')



    
    
    
    def test(self, save_parameters='no'):    
        """
        Parameters
        ----------
        save_parameters: indicates if the user wants to save the values of the estimated parameters (default is 'no')
        """
        if save_parameters == 'no':

            data = MyDatasetFolder(self.path_test, self.K, self.N, self.size)
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
            # Evaluation mode
            self.model.eval()
            
            batch_times = []
            isi_scores = []

            
            with torch.no_grad():
                for minibatch in tqdm(loader, file=sys.stdout):
                    
                    print("\n")
                    #print("minibatch: ", minibatch[0].shape)
                    X, A = minibatch

                    if torch.isinf(X).any() or torch.isinf(A).any():
                        print(f"NaNs detected in input data at batch")
                        print(f"X: {X}")
                        print(f"A: {A}")
                        return
                    
                    start_time = time.time()

                    W_predicted, _ = self.model(X, A, self.mode)

                    end_time = time.time()
                    batch_time = end_time - start_time

                    if torch.isnan(W_predicted).any():
                        print(f"NaNs detected in model output at batch")
                        print("W predicted",W_predicted)
                        return
                    
                    # Calculer le score ISI pour le lot actuel
                    isi_score = self.loss_fun(W_predicted, A)
                    print(f'ISI Score for current batch: {isi_score:.4f}')
                    print(f'Time for current batch: {batch_time:.5f} seconds')
                    batch_times.append(batch_time)
                    isi_scores.append(isi_score * len(X))

                
                
                self.model.isi_scores = [score / 100 for score in self.model.isi_scores]



                # Calculer le score ISI moyen
                print("sum isi scores: ", sum(isi_scores))
                print("len loader.dataset: ", len(loader.dataset))
                mean_isi_score = sum(isi_scores) / len(loader.dataset)

                # Calculer le temps moyen par donnée et par lot
                total_time = sum(batch_times)
                self.test_mean_time_per_data = total_time / len(loader.dataset)
                self.test_mean_time_per_batch = total_time / len(loader)
                print("\n")
                print("-----------------")
                print("Test Evaluation")
                print("-----------------")


                print(f'Test ISI Score: {mean_isi_score:.4f}')
                #print(f'Test Time: {mean_time:.5f} seconds')
            
        else :
            pass
    

        


                





