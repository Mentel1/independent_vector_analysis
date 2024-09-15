import os
import torch
import shutil
from data import *

def save_dataset(base_path, mode, size, T, K, N, epsilon=1, rho_bounds=[0.4, 0.6], lambda_=0.25):
    assert mode in ['train', 'test', 'val'], f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'."
    
    # Créer le chemin pour le sous-dossier spécifique au mode
    mode_path = os.path.join(base_path, mode, f"K_{K}_N_{N}",cases)

    # Check if the folder already exists
    if os.path.exists(mode_path):
        # If it exists, delete it and its contents
        shutil.rmtree(mode_path)
        print(f"Existing folder deleted: {mode_path}")
    
    # Create the folder again
    os.makedirs(mode_path, exist_ok=True)
    print(f"Folder created: {mode_path}")

    saved_count = 0

    while saved_count < size:
        # Generate data X and A using generate_whitened_problem
        X, A = generate_whitened_problem(T, K, N, epsilon, rho_bounds, lambda_)

        # Check for NaNs in X and A
        if torch.isnan(X).any() or torch.isnan(A).any():
            print(f"NaNs detected in generated data pair {saved_count}. Regenerating data.")
            continue
        
        # Save tuple (X, A) into a single file
        torch.save((X, A), os.path.join(mode_path, f"X_A_{saved_count}.pt"))
        saved_count += 1



# Exemple de génération de données et sauvegarde
base_path = "Unrolled-TITAN/TITAN_Unrolled/Datasets"
cases = 'ABCD'
path = os.path.join(base_path, f"Cases_{cases}")
size = 1000
K_values = [20]
N_values = [20]
T = 10000  
mode = 'train'

for K in K_values:
    for N in N_values:
        save_dataset(path, mode, size, T, K, N)
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
