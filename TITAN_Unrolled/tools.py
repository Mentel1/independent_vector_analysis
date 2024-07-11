import torch


def sym(A):
    if A.ndim == 2:
        return (A.transpose(0, 1) + A) / 2
    elif A.ndim == 3:
        return (A + torch.moveaxis(A, 1, 2)) / 2
    else:
        return (A + torch.moveaxis(A, 1, 2)) / 2
                


def cov_X(X):
    B,_, T, _ = X.size()
    Rx = torch.einsum('bNTK,bMTJ->bKJNM', X, X) / T
    return Rx


def covid(X):
    """
    Computes the covariance matrix of the input tensor X.
    Parameters
    ----------
        X (torch.FloatTensor): input tensor, size B x N x T x K
    Returns
    -------
        Rx (torch.FloatTensor): covariance matrix, size B x K x K x N x N
    """
    B, N, T, K = X.size()
    
    # Debugging prints
    print(f"Input X shape: {X.shape}")
    

    
    try:
        Rx = torch.einsum('bntk,bmtj->bkjmn', X, X) / T
        print("Successfully computed covariance matrix")
    except RuntimeError as e:
        print(f"Error in einsum operation: {e}")
        return None
    
    return Rx



def spectral_norm(M):
    if M.dim() == 2:
        return torch.linalg.norm(M, ord=2)
    else:
        return torch.max(torch.linalg.norm(M, ord=2,dim=(1,2)))
    

def spectral_norm_extracted(Rx, K, N):
    # Rx is expected to have shape (B, K*K*N, N) for batched input
    B = Rx.shape[0]
    # Reshape Rx to (B, K, K*N, N)
    Rx_reshaped = torch.reshape(Rx, (B, K, K*N, N))
    # Compute the 2-norm over dimensions (2, 3)
    norms = torch.norm(Rx_reshaped, p=2, dim=(2, 3))
    # Return the maximum norm for each batch
    return torch.max(norms, dim=1).values


def smallest_singular_value(C):
    _, s, _ = torch.svd(C.permute(2, 0, 1))
    return torch.min(s)

def block_diag(W):
    N, N, K = W.size()
    W_bd = torch.zeros(K, K*N, N,device='cuda')
    for k in range(K):
        W_bd[k, k*N:(k+1)*N, :] = W[:, :, k].t()
    return W_bd

def blocks_to_full(W_blocks, K, N):
    W_full = torch.zeros(N, N, K,device='cuda')
    for k, W_k in enumerate(W_blocks):
        begin = 0
        for W_kl in W_k:
            n_l = W_kl.size(0)
            W_full[begin:begin+n_l, begin:begin+n_l, k] = W_kl
            begin += n_l
    return W_full

def full_to_blocks(W_full, idx_W, K):
    W_blocks = []
    for k in range(K):
        W_k = []
        L_k = len(idx_W[k])
        for l in range(L_k):
            W_kl = W_full[idx_W[k][l], idx_W[k][l], k]
            W_k.append(W_kl)
        W_blocks.append(W_k)
    return W_blocks

def lipschitz(C,rho_Rx):
    return spectral_norm(C)*rho_Rx


def joint_isi_batch(W, A):
    batch_size, N, _, _ = W.shape
    G_bar = torch.sum(torch.abs(torch.einsum('bnNk,bNvk->bnvk', W, A)), dim=3)
    score = (torch.sum(torch.sum(G_bar / torch.max(G_bar, dim=1, keepdim=True)[0], dim=1), dim=0) +
             torch.sum(torch.sum(G_bar.transpose(1, 2) / torch.max(G_bar.transpose(1, 2), dim=1, keepdim=True)[0], dim=1), dim=0))
    return score / (2 * N * (N - 1))


def joint_isi(W, A):
    N, _, _ = W.shape
    G_bar = torch.sum(torch.abs(torch.einsum('nNk,Nvk->nvk', W, A)), dim=2)
    score = (torch.sum(torch.sum(G_bar / torch.max(G_bar, dim=0)[0], dim=0) - 1) +
             torch.sum(torch.sum(G_bar.t() / torch.max(G_bar.t(), dim=0)[0], dim=0) - 1))
    return score / (2 * N * (N - 1))



def decrease(cost, verbose=0):
    accr = torch.tensor(cost[:-1]) - torch.tensor(cost[1:])
    if torch.all(accr >= 0):
        return True
    else:
        if verbose >= 1:
            for i in range(len(accr)):
                if accr[i] < 0:
                    print("increase at index :", i)
                    print("an increase of :", -accr[i])
                    break
        return False
    
def diff_criteria(A, B, mode='full'):
    if mode == 'full':
        if A.shape != B.shape:
            raise ValueError("A and B must be of the same dimension")
        elif A.ndim < 2 or A.ndim > 3:
            raise ValueError("Only tensors of order 2 or 3 are accepted")
        res = 0
        D = A - B

        max_norm = torch.max(torch.sum(D ** 2, dim=1))
        return max_norm / (2 * A.size(0))


    elif mode == 'blocks':
        res = 0
        if len(A) != len(B):
            raise ValueError("A and B must be of the same dimension")
        for k, A_k in enumerate(A):
            B_k = B[k]
            if len(A_k) != len(B_k):
                raise ValueError("A_k and B_k must have the same blocks")
            for l, A_kl in enumerate(A_k):
                B_kl = B_k[l]
                if A_kl.shape != B_kl.shape:
                    raise ValueError("A_k and B_k must have the same blocks")
                N_kl, _ = A_kl.shape
                for n in range(N_kl):
                    res = max(res, torch.dot(A_kl[n, :], B_kl[n, :]) / (2 * N_kl))
        return res

        

