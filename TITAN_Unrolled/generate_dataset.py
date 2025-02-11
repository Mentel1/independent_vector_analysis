import os
import torch
import shutil
from data import *

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
        X, A = generate_whitened_problem(T, K, N, epsilon, rho_bounds, lambda_)

        # Check for NaNs in X and A
        if torch.isnan(X).any() or torch.isnan(A).any():
            print(f"NaNs detected in generated data pair {saved_count}. Regenerating data.")
            continue
        
        # Save tuple (X, A) into a single file
        torch.save((X, A), os.path.join(save_path, f"X_A_{saved_count}.pt"))
        saved_count += 1



# Exemple de génération de données et sauvegarde
base_path = "Unrolled-TITAN/TITAN_Unrolled/Datasets"
mode = 'train'
cases = 'easy cases'
path = os.path.join(base_path, mode, cases)
size = 1000
K_values = [10]
N_values = [10]
T = 10000  

for K in K_values:
    for N in N_values:
        save_dataset(path, cases, size, T, K, N)
        print(f"Dataset généré et sauvegardé avec succès pour K={K} et N={N} !")


print("Datasets générés et sauvegardés avec succès !")







""" lambda_1 = 0.04
lambda_2 = 0.25
rho_bounds_1 = [0.2,0.3]
rho_bounds_2 = [0.6,0.7]
rhos = [rho_bounds_1,rho_bounds_2]
lambdas = [lambda_1,lambda_2]

metaparameters_multiparam = get_metaparameters(rhos,lambdas)
metaparameters_titles_multiparam = ['Case A','Case B','Case C','Case D']
 """
