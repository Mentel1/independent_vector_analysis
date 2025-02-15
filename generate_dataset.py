import os
import torch
import shutil
from algorithms.problem_simulation import *
from algorithms.helpers_iva import *

def save_dataset(base_path, cases, size, T, K, N, epsilon=1, rho_bounds=[0.4, 0.6], lambda_=0.25):
    
    # Créer le chemin pour le sous-dossier spécifique au mode
    save_path = os.path.join(base_path, f"K_{K}_N_{N}")

    # Check if the folder already exists
    if os.path.exists(save_path):
        # If it exists, delete it and its contents
        shutil.rmtree(save_path)
        print(f"Existing folder deleted: {save_path}")

    # Create the folder again
    os.makedirs(save_path, exist_ok=True)
    print(f"Folder created: {save_path}")

    saved_count = 0

    if cases == 'easy cases':
        rho_bounds = [0.2, 0.7]
        lambda_ = 0.25

    elif cases == 'hard cases':
        rho_bounds = [0.2, 0.7]
        lambda_ = 0.04



    while saved_count < size:
        # Generate data X and A using generate_whitened_problem
        Sigma = make_Sigma(K,N,rank=K+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
        S = make_S(Sigma,T)
        A = make_Sigma(N,K,rank=N+10,epsilon=epsilon,rho_bounds=rho_bounds,lambda_=lambda_,seed=None,normalize=False)
        X = make_X(S,A)
        # translate all tensors in pytorch
        S = torch.tensor(S)
        X = torch.tensor(X) 
        A = torch.tensor(A)
        X_,U = whiten_data_torch(X)
        A_ = torch.einsum('nNk,Nvk->nvk',U,A)
        X,A = X_,A_
        # Check for NaNs in X and A
        if torch.isnan(X).any() or torch.isnan(A).any():
            print(f"NaNs detected in generated data pair {saved_count}. Regenerating data.")
            continue
        
        # Save tuple (X, A) into a single file
        torch.save((X, A), os.path.join(save_path, f"X_A_{saved_count}.pt"))
        saved_count += 1
        if saved_count % 100 == 0:
            print(f"Saved {saved_count} datasets for K={K} and N={N}.")


# Exemple de génération de données et sauvegarde
base_path = "Datasets"
mode = 'train'
cases = 'easy cases'
path = os.path.join(base_path, mode, cases)
size = 1000
K_values = [20]
N_values = [20]
T = 10000  

for K in K_values:
    for N in N_values:
        save_dataset(path, cases, size, T, K, N)
        print(f"Dataset généré et sauvegardé avec succès pour K={K} et N={N} !")


print("Datasets générés et sauvegardés avec succès !")
