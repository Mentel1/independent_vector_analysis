#     Copyright (c) <2021> <University of Paderborn>
#     Signal and System Theory Group, Univ. of Paderborn, https://sst-group.org/
#     https://github.com/SSTGroup/independent_vector_analysis
#
#     Permission is hereby granted, free of charge, to any person
#     obtaining a copy of this software and associated documentation
#     files (the "Software"), to deal in the Software without restriction,
#     including without limitation the rights to use, copy, modify and
#     merge the Software, subject to the following conditions:
#
#     1.) The Software is used for non-commercial research and
#        education purposes.
#
#     2.) The above copyright notice and this permission notice shall be
#        included in all copies or substantial portions of the Software.
#
#     3.) Publication, Distribution, Sublicensing, and/or Selling of
#        copies or parts of the Software requires special agreements
#        with the University of Paderborn and is in general not permitted.
#
#     4.) Modifications or contributions to the software must be
#        published under this license. The University of Paderborn
#        is granted the non-exclusive right to publish modifications
#        or contributions in future versions of the Software free of charge.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#     OTHER DEALINGS IN THE SOFTWARE.
#
#     Persons using the Software are encouraged to notify the
#     Signal and System Theory Group at the University of Paderborn
#     about bugs. Please reference the Software in your publications
#     if it was used for them.

import torch
import numpy as np
import scipy as sc
import scipy.linalg
from scipy.special import gamma

def _normalize_column_vectors(x):
    return x / x.norm(dim=0)

def _decouple_trick_numpy(W, n, Q=None, R=None):
        # get dimensions
    M, N, K = W.shape
    if M != N:
        raise AssertionError('Assuming W is square matrix.')
    H = np.zeros((N, K), dtype=W.dtype)

    # use QR recursive method
    if n == 0:
        Q_ = np.zeros((N, N, K), dtype=W.dtype)
        R_ = np.zeros((N, N - 1, K), dtype=W.dtype)
    else:
        Q_ = np.copy(Q)
        R_ = np.copy(R)

    for k in range(K):
        if n == 0:
            W_tilde = W[1:N, :, k]
            Q_[:, :, k], R_[:, :, k] = np.linalg.qr(np.conj(W_tilde.T), mode='complete')
        else:
            n_last = n - 1
            e_last = np.zeros(N - 1, dtype=W.dtype)
            e_last[n_last] = 1

            # e_last.shape and R.shape[1], W.shape[1] and Q.shape[1] must be equal
            Q_[:, :, k], R_[:, :, k] = sc.linalg.qr_update(Q_[:, :, k], R_[:, :, k],
                                                           -np.conj(W[n, :, k]), e_last)
            Q_[:, :, k], R_[:, :, k] = sc.linalg.qr_update(Q_[:, :, k], R_[:, :, k],
                                                           np.conj(W[n_last, :, k]), e_last)

        H[:, k] = Q_[:, -1, k]

    return H, Q_, R_

def _decouple_trick_torch(W, n, Q=None, R=None):
    W_ = W.numpy()
    # if Q is not None:
    #     Q.numpy
    # if R is not None:
    #     R.numpy
    H, Q_, R_ = _decouple_trick_numpy(W_,n,Q,R)
    H = torch.from_numpy(H)
    Q_ = torch.from_numpy(Q_)
    R_= torch.from_numpy(R_)

    return H, Q_, R_

# def fast_decouple_trick_numpy(W, Q=None, R=None):
#         # get dimensions
#     M, N, K = W.shape
#     if M != N:
#         raise AssertionError('Assuming W is square matrix.')
#     H = np.zeros((N, K), dtype=W.dtype)

#     # use QR recursive method
#     if n == 0:
#         Q_ = np.zeros((N, N, K), dtype=W.dtype)
#         R_ = np.zeros((N, N - 1, K), dtype=W.dtype)
#     else:
#         Q_ = np.copy(Q)
#         R_ = np.copy(R)

#     for k in range(K):
#         if n == 0:
#             W_tilde = W[1:N, :, k]
#             Q_[:, :, k], R_[:, :, k] = np.linalg.qr(np.conj(W_tilde.T), mode='complete')
#         else:
#             n_last = n - 1
#             e_last = np.zeros(N - 1, dtype=W.dtype)
#             e_last[n_last] = 1

#             # e_last.shape and R.shape[1], W.shape[1] and Q.shape[1] must be equal
#             Q_[:, :, k], R_[:, :, k] = sc.linalg.qr_update(Q_[:, :, k], R_[:, :, k],
#                                                            -np.conj(W[n, :, k]), e_last)
#             Q_[:, :, k], R_[:, :, k] = sc.linalg.qr_update(Q_[:, :, k], R_[:, :, k],
#                                                            np.conj(W[n_last, :, k]), e_last)

#         H[:, k] = Q_[:, -1, k]

#     return H, Q_, R_

def _bss_isi_torch(W, A, s=None):
    """
    Calculate measure of quality of separation for blind source separation algorithms.
    Model:   x = A @ s,   y = W @ x = W @ A @ s

    Parameters
    ----------
    W : torch.Tensor
        demixing matrix of dimensions N x N x K or M x p

    A : torch.Tensor
        true mixing matrix of dimension N x N x K or p x N

    s : torch.Tensor, optional
        true sources of dimension N x T x K or N x T

    (p: #sensors, N: #sources, M: #estimatedsources, K: #datasets, T: #samples)

    Returns
    -------
    avg_isi : float
        avg_isi=0 is optimal.
        Normalized performance index is given in Choi, S. Cichocki, A. Zhang, L. & Amari, S.
        Approximate maximum likelihood source separation using the natural gradient
        Wireless Communications, 2001. (SPAWC '01). 2001 IEEE Third Workshop on Signal
        Processing Advances in, 2001, 235-238.

    joint_isi : float, optional
        joint_isi = 0 is optimal. Only calculated if there are at least 2 datasets, otherwise
        np.nan.
        Normalized joint performance index given in Anderson, Matthew, Tuelay Adali, and
        Xi-Lin Li. "Joint blind source separation with multivariate Gaussian model: Algorithms and
        performance analysis." IEEE Transactions on Signal Processing 60.4 (2011): 1672-1683.

    Notes
    -----
    W is the estimated demixing matrix and A is the true mixing matrix.  It should be noted
    that rows of the mixing matrix should be scaled by the necessary constants such that each
    source has unit variance and accordingly each row of the demixing matrix should be
    scaled such that each estimated source has unit variance.

    Note that A is p x N, where p is the number of sensors and N is the number of signals
    and W is M x p, where M is the number of estimated signals.  Ideally M=N but this is not
    guaranteed.  So if M > N, the algorithm has estimated more sources than it "should", and
    if M < N the algorithm has not found all of the sources.

    Code is converted to PyTorch by [Your Name]
    """

    # generalized permutation invariant flag (default=False), only used when s is None
    gen_perm_inv_flag = False

    if W.ndim == 2 and A.ndim == 2:
        if s is None:
            # Traditional metric, user provided W & A separately
            G = torch.matmul(W, A)
            M, N = G.shape
            G_abs = torch.abs(G)
            if gen_perm_inv_flag:
                # normalization by row
                G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]
        else:
            # Equalize energy associated with each estimated source and true source.
            y = torch.matmul(torch.matmul(W, A), s)
            # Standard deviation calculated with n-1
            D = torch.diag(
                1 / torch.std(s, dim=1, unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
            U = torch.diag(
                1 / torch.std(y, dim=1, unbiased=True))  # y_norm = U @ y, where y_norm has unit variance

            # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
            G = torch.matmul(torch.matmul(U, W), torch.solve(A.t(), D.t())[0].t())  # A @ np.linalg.inv(D)
            M, N = G.shape
            G_abs = torch.abs(G)

        avg_isi = 0
        for m in range(M):
            avg_isi += torch.sum(G_abs[m, :]) / torch.max(G_abs[m, :]) - 1

        for n in range(N):
            avg_isi += torch.sum(G_abs[:, n]) / torch.max(G_abs[:, n]) - 1

        avg_isi /= (2 * N * (N - 1))

        return avg_isi, torch.nan

    elif W.ndim == 3 and A.ndim == 3:
        # IVA/GroupICA/MCCA Metrics
        # For this we want to average over the K groups as well as provide the additional
        # measure of solution to local permutation ambiguity (achieved by averaging the K
        # demixing-mixing matrices and then computing the ISI of this matrix).

        N, M, K = W.shape
        if M != N:
            raise AssertionError('This more general case has not been considered here.')

        avg_isi = 0
        G_abs_total = torch.zeros((N, N))
        G = torch.zeros((N, N, K), dtype=W.dtype)
        for k in range(K):
            if s is None:
                # Traditional metric, user provided W & A separately
                G_k = torch.matmul(W[:, :, k], A[:, :, k])
                G_abs = torch.abs(G_k)
                if gen_perm_inv_flag:
                    # normalization by row
                    G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]
            else:

                # Equalize energy associated with each estimated source and true source.
                # Standard deviation calculated with n-1
                y_k = torch.matmul(torch.matmul(W[:, :, k], A[:, :, k]), s[:, :, k])
                D_k = torch.diag(1 / torch.std(s[:, :, k], dim=1,
                                               unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
                U_k = torch.diag(1 / torch.std(y_k, dim=1,
                                               unbiased=True))  # y_norm = U @ y, where y_norm has unit variance
                # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
                G_k = torch.matmul(torch.matmul(U_k, W[:, :, k]), torch.solve(A[:, :, k].t(), D_k.t())[0].t())  # A @ np.linalg.inv(D)
                G_abs = torch.abs(G_k)

            G[:, :, k] = G_k

            G_abs_total += G_abs

            for n in range(N):
                avg_isi += torch.sum(G_abs[n, :]) / torch.max(G_abs[n, :]) - 1

            for m in range(N):
                avg_isi += torch.sum(G_abs[:, m]) / torch.max(G_abs[:, m]) - 1

        avg_isi /= (2 * N * (N - 1) * K)

        G_abs = torch.copy(G_abs_total)
        if gen_perm_inv_flag:
            # normalization by row
            G_abs /= torch.max(G_abs, dim=1, keepdim=True)[0]

        joint_isi = 0
        for n in range(N):
            joint_isi += torch.sum(G_abs[n, :]) / torch.max(G_abs[n, :]) - 1

        for m in range(N):
            joint_isi += torch.sum(G_abs[:, m]) / torch.max(G_abs[:, m]) - 1

        joint_isi /= (2 * N * (N - 1))

        return avg_isi, joint_isi
    else:
        raise AssertionError('All inputs must be of either dimension 2 or 3')
    
def _bss_isi(W, A, s=None):

    # generalized permutation invariant flag (default=False), only used when s is None
    gen_perm_inv_flag = False

    if W.ndim == 2 and A.ndim == 2:
        if s is None:
            # Traditional metric, user provided W & A separately
            G = np.matmul(W, A)
            M, N = G.shape
            G_abs = np.abs(G)
            if gen_perm_inv_flag:
                # normalization by row
                G_abs /= np.max(G_abs, dim=1, keepdim=True)[0]
        else:
            # Equalize energy associated with each estimated source and true source.
            y = np.matmul(np.matmul(W, A), s)
            # Standard deviation calculated with n-1
            D = np.diag(
                1 / np.std(s, dim=1, unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
            U = np.diag(
                1 / np.std(y, dim=1, unbiased=True))  # y_norm = U @ y, where y_norm has unit variance

            # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
            G = np.matmul(np.matmul(U, W), np.linalg.solve(A.t(), D.t())[0].t())  # A @ np.linalg.inv(D)
            M, N = G.shape
            G_abs = np.abs(G)

        avg_isi = 0
        for m in range(M):
            avg_isi += np.sum(G_abs[m, :]) / np.max(G_abs[m, :]) - 1

        for n in range(N):
            avg_isi += np.sum(G_abs[:, n]) / np.max(G_abs[:, n]) - 1

        avg_isi /= (2 * N * (N - 1))

        return avg_isi, np.nan

    elif W.ndim == 3 and A.ndim == 3:
        # IVA/GroupICA/MCCA Metrics
        # For this we want to average over the K groups as well as provide the additional
        # measure of solution to local permutation ambiguity (achieved by averaging the K
        # demixing-mixing matrices and then computing the ISI of this matrix).

        N, M, K = W.shape
        if M != N:
            raise AssertionError('This more general case has not been considered here.')

        avg_isi = 0
        G_abs_total = np.zeros((N, N))
        G = np.zeros((N, N, K), dtype=W.dtype)
        for k in range(K):
            if s is None:
                # Traditional metric, user provided W & A separately
                G_k = np.matmul(W[:, :, k], A[:, :, k])
                G_abs = np.abs(G_k)
                if gen_perm_inv_flag:
                    # normalization by row
                    G_abs /= np.max(G_abs, dim=1, keepdim=True)[0]
            else:

                # Equalize energy associated with each estimated source and true source.
                # Standard deviation calculated with n-1
                y_k = np.matmul(np.matmul(W[:, :, k], A[:, :, k]), s[:, :, k])
                D_k = np.diag(1 / np.std(s[:, :, k], dim=1,
                                               unbiased=True))  # s_norm = D @ s, where s_norm has unit variance
                U_k = np.diag(1 / np.std(y_k, dim=1,
                                               unbiased=True))  # y_norm = U @ y, where y_norm has unit variance
                # Thus: y_norm = U @ W @ A @ np.linalg.inv(D) @ s_norm = G @ s_norm, and
                G_k = np.matmul(np.matmul(U_k, W[:, :, k]), np.linalg.solve(A[:, :, k].t(), D_k.t())[0].t())  # A @ np.linalg.inv(D)
                G_abs = np.abs(G_k)

            G[:, :, k] = G_k

            G_abs_total += G_abs

            for n in range(N):
                avg_isi += np.sum(G_abs[n, :]) / np.max(G_abs[n, :]) - 1

            for m in range(N):
                avg_isi += np.sum(G_abs[:, m]) / np.max(G_abs[:, m]) - 1

        avg_isi /= (2 * N * (N - 1) * K)

        G_abs = np.copy(G_abs_total)
        if gen_perm_inv_flag:
            # normalization by row
            G_abs /= np.max(G_abs, dim=1, keepdim=True)[0]

        joint_isi = 0
        for n in range(N):
            joint_isi += np.sum(G_abs[n, :]) / np.max(G_abs[n, :]) - 1

        for m in range(N):
            joint_isi += np.sum(G_abs[:, m]) / np.max(G_abs[:, m]) - 1

        joint_isi /= (2 * N * (N - 1))

        return avg_isi, joint_isi
    else:
        raise AssertionError('All inputs must be of either dimension 2 or 3')

def whiten_data_numpy(x, dim_red=None):
    """
    Whitens the data vector so that E{z z.T} = I, where z = V @ x.
    Optionally, a dimension reduction can be performed.


    Parameters
    ----------
    x : np.ndarray
        data vector of dimensions N x T x K (#sources x #samples x #datasets) or N x T

    dim_red : int, optional
        dimension to which the data should be reduced


    Returns
    -------
    z : np.ndarray
        whitened data of dimension N x T x K or N x T

    V : np.ndarray
        whitening transformation of dimension N x N x K or N x N

    """
    if dim_red is None:
        dim_red = x.shape[0]

    if x.ndim == 2:
        N, T = x.shape

        # Step 1. Center the data.
        x_zm = x - np.mean(x, axis=1, keepdims=True)

        # Step 2. Form MLE of data covariance.
        covar = np.cov(x_zm, ddof=0)

        # Step 3. Eigen decomposition of covariance.
        eigval, eigvec = np.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = np.flip(eigval)
        eigvec = np.fliplr(eigvec)

        # Step 4. Forming whitening transformation.
        V = np.einsum('n,Nn -> nN', 1 / np.sqrt(eigval[0:dim_red]), np.conj(eigvec[:, 0:dim_red]))

        # Step 5. Form whitened data
        z = V @ x_zm

    else:
        N, T, K = x.shape

        eigval = np.zeros((N, K), dtype=x.dtype)
        eigvec = np.zeros((N, N, K), dtype=x.dtype)
        x_zm = np.zeros_like(x)

        for k in range(K):
            # Step 1. Center the data.
            x_zm[:, :, k] = x[:, :, k] - np.mean(x[:, :, k], axis=1, keepdims=True)

            # Step 2. Form MLE of data covariance.
            covar = np.cov(x_zm[:, :, k], ddof=0)

            # Step 3. Eigen decomposition of covariance.
            eigval[:, k], eigvec[:, :, k] = np.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = np.flipud(eigval)
        eigvec = np.flip(eigvec, 1)

        # efficient implementation of
        #     V[:, :, k] = np.linalg.solve(np.diag(np.sqrt(eigval)), np.conj(eigvec.T))
        #     z[:, :, k] = V[:, :, k] @ x_k

        # Step 4. Forming whitening transformation.
        V = np.einsum('nk,Nnk -> nNk', 1 / np.sqrt(eigval[0:dim_red, :]),
np.conj(eigvec[:, 0:dim_red, :]))

        # Step 5. Form whitened data
        z = np.einsum('nNk, Nvk-> nvk', V, x_zm)
    return z, V

def whiten_data_torch(x, dim_red=None):
    if dim_red is None:
        dim_red = x.shape[0]

    if x.ndim == 2:
        N, T = x.shape

        # Step 1. Center the data.
        x_zm = x - torch.mean(x, dim=1, keepdim=True)

        # Step 2. Form MLE of data covariance.
        covar = torch.matmul(x_zm, x_zm.transpose(0, 1)) / T

        # Step 3. Eigen decomposition of covariance.
        eigval, eigvec = torch.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = torch.flip(eigval, dims=[0])
        eigvec = torch.fliplr(eigvec)

        # Step 4. Forming whitening transformation.
        V = torch.einsum('n,Nn -> nN', 1 / torch.sqrt(eigval[0:dim_red]), torch.conj(eigvec[:, 0:dim_red]))

        # Step 5. Form whitened data
        z = torch.matmul(V, x_zm)

    else:
        N, T, K = x.shape

        eigval = torch.zeros((N, K), dtype=x.dtype)
        eigvec = torch.zeros((N, N, K), dtype=x.dtype)
        x_zm = torch.zeros_like(x)

        for k in range(K):
            # Step 1. Center the data.
            x_zm[:, :, k] = x[:, :, k] - torch.mean(x[:, :, k], dim=1, keepdim=True)

            # Step 2. Form MLE of data covariance.
            covar = torch.matmul(x_zm[:, :, k], x_zm[:, :, k].transpose(0, 1)) / T

            # Step 3. Eigen decomposition of covariance.
            eigval[:, k], eigvec[:, :, k] = torch.linalg.eigh(covar)

        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = torch.flipud(eigval)
        eigvec = torch.flip(eigvec, dims=[1])

        # Step 4. Forming whitening transformation.
        V = torch.einsum('nk,Nnk -> nNk', 1 / torch.sqrt(eigval[0:dim_red, :]),
                      torch.conj(eigvec[:, 0:dim_red, :]))

        # Step 5. Form whitened data
        z = torch.einsum('nNk, Nvk-> nvk', V, x_zm)
    return z, V

def fast_whiten_data_torch(x, dim_red=None):
    if dim_red is None:
        dim_red = x.shape[0]

    if x.ndim == 2:
        N, T = x.shape
        x_zm = x - torch.mean(x, dim=1, keepdim=True)
        covar = torch.matmul(x_zm, x_zm.transpose(0, 1)) / T
        eigval, eigvec = torch.linalg.eigh(covar)
        eigval = torch.flip(eigval, dims=[0])
        eigvec = torch.fliplr(eigvec)
        V = torch.einsum('n,Nn -> nN', 1 / torch.sqrt(eigval[0:dim_red]), torch.conj(eigvec[:, 0:dim_red]))
        z = torch.matmul(V, x_zm)

    else:
        N, T, K = x.shape
        x_zm = x - torch.mean(x, dim=1, keepdim=True)
        covar = torch.einsum('ntk,mtk->knm',x_zm,x_zm)/T
        eigval, eigvec = torch.linalg.eigh(covar)
        eigval = eigval.transpose(0,1)
        eigvec = eigvec.permute(1,2,0)
        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = torch.flipud(eigval)
        eigvec = torch.flip(eigvec, dims=[1])

        # Step 4. Forming whitening transformation.
        V = torch.einsum('nk,Nnk -> nNk', 1 / torch.sqrt(eigval[0:dim_red, :]),
                      torch.conj(eigvec[:, 0:dim_red, :]))

        # Step 5. Form whitened data
        z = torch.einsum('nNk, Nvk-> nvk', V, x_zm)
    return z, V

def fast_whiten_data_numpy(x, dim_red=None):
    if dim_red is None:
        dim_red = x.shape[0]

    if x.ndim == 2:
        N, T = x.shape
        x_zm = x - np.mean(x,axis=1,keepdims=True)
        covar = np.matmul(x_zm, x_zm.transpose(0, 1)) / T
        eigval, eigvec = np.linalg.eigh(covar)
        eigval = np.flip(eigval, dims=[0])
        eigvec = np.fliplr(eigvec)
        V = np.einsum('n,Nn -> nN', 1 / np.sqrt(eigval[0:dim_red]), np.conj(eigvec[:, 0:dim_red]))
        z = np.matmul(V, x_zm)

    else:
        N, T, K = x.shape
        x_zm = x - np.mean(x,axis=1,keepdims=True)
        covar = np.einsum('ntk,mtk->knm',x_zm,x_zm)/T
        eigval, eigvec = np.linalg.eigh(covar)
        eigval = np.transpose(eigval,(1,0))
        eigvec = np.transpose(eigvec, (1, 2, 0))
        # sort eigenvalues and corresponding eigenvectors in descending order
        eigval = np.flipud(eigval)
        eigvec = np.flip(eigvec, 1)

        # Step 4. Forming whitening transformation.
        V = np.einsum('nk,Nnk -> nNk', 1 / np.sqrt(eigval[0:dim_red, :]),
                      eigvec[:, 0:dim_red, :])

        # Step 5. Form whitened data
        z = np.einsum('nNk, Nvk-> nvk', V, x_zm)
    return z, V

def _comp_l_sos_cost(W, Y, const_log=None, scale_sources=False):
    N, T, K = Y.shape
    cost = 0
    xi_shape = K + 1  # .. Modified by Suchita Bhinge... September 14 2018
    if const_log is None:
        const_Ck = 0.5 * gamma(K / 2) * xi_shape ** (K / 2) / (torch.pi ** (K / 2) * gamma(K))
        const_log = -torch.log(const_Ck)

    if scale_sources:
        for k in range(K):
            ypower = torch.diag(1 / T * torch.matmul(Y[:, :, k], torch.conj(Y[:, :, k].t())))
            W[:, :, k] /= torch.sqrt(ypower)[:, None]

    for k in range(K):
        cost -= torch.log(torch.abs(sc.det(W[:, :, k])))

    for n in range(N):
        yn = Y[n, :, :].t()  # K x T
        CovN = 1 / T * torch.matmul(yn, torch.conj(yn.t()))
        gip = torch.sum(torch.conj(yn) * sc.solve(CovN, yn), dim=0)
        dcost = (const_log + 0.5 * torch.log(sc.det(CovN))) + xi_shape ** 0.5 * torch.mean(
            gip ** 0.5)  # .. Modified by Suchita Bhinge... September 14 2018
        cost += dcost
    y = Y.clone()

    return cost, y

def _resort_scvs_torch(W, R_xx, whiten=False, V=None, complex_valued=False, circular=False, P_xx=None):
    """
    Resort order of SCVs: Order the components from most to least ill-conditioned
    """

    N, _, K = W.shape

    # First, compute the data covariance matrices (by undoing any whitening)
    if whiten:
        for k1 in range(K):
            for k2 in range(k1, K):
                R_xx[:, :, k1, k2] = torch.linalg.solve(R_xx[:, :, k1, k2], V[:, :, k2])[0] @ torch.linalg.solve(R_xx[:, :, k1, k2].mT, V[:, :, k1])[0]
                R_xx[:, :, k2, k1] = R_xx[:, :, k1, k2].conj().mT  # R_xx is Hermitian

    # Second, compute the determinant of the SCVs
    if complex_valued:
        detSCV = torch.zeros(N, dtype=torch.complex64)
        Sigma_N = torch.zeros((K, K, N), dtype=torch.complex64)
    else:
        detSCV = torch.zeros(N)
        Sigma_N = torch.zeros((K, K, N))

    for n in range(N):
        # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
        if complex_valued:
            Sigma_n = torch.zeros((K, K), dtype=torch.complex64)
        else:
            Sigma_n = torch.zeros((K, K))

        for k1 in range(K):
            for k2 in range(k1, K):
                Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ W[n, :, k2].conj()
                Sigma_n[k2, k1] = Sigma_n[k1, k2].conj()  # sigma_n is Hermitian
        Sigma_N[:, :, n] = Sigma_n

        if complex_valued and not circular:
            Sigma_P_n = torch.zeros((K, K), dtype=torch.complex64)  # pseudo = 1/T * Y_n @ Y_n.T
            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_P_n[k1, k2] = W[n, :, k1] @ P_xx[:, :, k1, k2] @ W[n, :, k2]
                    Sigma_P_n[k2, k1] = Sigma_P_n[k1, k2]  # sigma_P_n is symmetric
            detSCV[n] = torch.det(torch.cat([torch.cat([Sigma_n, Sigma_P_n], dim=-1),
                                             torch.cat([Sigma_P_n.conj().t(), Sigma_n.conj()], dim=-1)], dim=-2))
        else:
            detSCV[n] = torch.det(Sigma_n)

    # Third, sort and apply
    isort = torch.argsort(detSCV)
    Sigma_N = Sigma_N[:, :, isort]
    for k in range(K):
        W[:, :, k] = W[isort, :, k]

    return W, Sigma_N

def _resort_scvs_numpy(W, R_xx, whiten=False, V=None, complex_valued=False, circular=False, P_xx=None):    
    
    N, _, K = W.shape

    # First, compute the data covariance matrices (by undoing any whitening)
    if whiten:
        for k1 in range(K):
            for k2 in range(k1, K):
                R_xx[:, :, k1, k2] = np.linalg.solve(V[:, :, k2],
                                                     np.linalg.solve(V[:, :, k1],
                                                                     R_xx[:, :, k1, k2]).T).T
                R_xx[:, :, k2, k1] = np.conj(R_xx[:, :, k1, k2].T)  # R_xx is Hermitian

    # Second, compute the determinant of the SCVs
    if complex_valued:
        detSCV = np.zeros(N, dtype=complex)
        Sigma_N = np.zeros((K, K, N), dtype=complex)
    else:
        detSCV = np.zeros(N)
        Sigma_N = np.zeros((K, K, N))

    for n in range(N):
        # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
        if complex_valued:
            Sigma_n = np.zeros((K, K), dtype=complex)
        else:
            Sigma_n = np.zeros((K, K))

        for k1 in range(K):
            for k2 in range(k1, K):
                Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ np.conj(W[n, :, k2])
                Sigma_n[k2, k1] = np.conj(Sigma_n[k1, k2])  # sigma_n is Hermitian
        Sigma_N[:, :, n] = Sigma_n

        if complex_valued and not circular:
            Sigma_P_n = np.zeros((K, K), dtype=complex)  # pseudo = 1/T * Y_n @ Y_n.T
            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_P_n[k1, k2] = W[n, :, k1] @ P_xx[:, :, k1, k2] @ W[n, :, k2]
                    Sigma_P_n[k2, k1] = Sigma_P_n[k1, k2]  # sigma_P_n is symmetric
            detSCV[n] = np.linalg.det(np.vstack([np.hstack([Sigma_n, Sigma_P_n]),
                                                 np.hstack(
                                                     [np.conj(Sigma_P_n.T), np.conj(Sigma_n)])]))
        else:
            detSCV[n] = np.linalg.det(Sigma_n)

    # Third, sort and apply
    isort = np.argsort(detSCV)
    Sigma_N = Sigma_N[:, :, isort]
    for k in range(K):
        W[:, :, k] = W[isort, :, k]

    return W, Sigma_N

