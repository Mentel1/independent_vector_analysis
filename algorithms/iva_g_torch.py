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
import time
from .helpers_iva import _normalize_column_vectors, _decouple_trick_torch, _bss_isi, whiten_data_torch, fast_whiten_data_torch, \
    _resort_scvs_torch
from .initializations import _jbss_sos, _cca


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def iva_g_torch(X, opt_approach='newton', complex_valued=False, circular=False, whiten=True,
          verbose=False, A=None, W_init=None, jdiag_initW=False, max_iter=1024,
          W_diff_stop=1e-6, alpha0=torch.tensor(1.0), return_W_change=False):
    """
    Implementation of all the second-order (Gaussian) independent vector analysis (IVA) algorithms.
    Namely real-valued and complex-valued with circular and non-circular using Newton, gradient,
    and quasi-Newton optimizations.
    For a general description of the algorithm and its relationship with others,
    see http://mlsp.umbc.edu/jointBSS_introduction.html.


    Parameters
    ----------
    X : torch.tensor
        data matrix of dimensions N x T x K.
        Data observations are from K data sets, i.e., X[k] = A[k] @ S[k], where A[k] is an N x N
        unknown invertible mixing matrix, and S[k] is N x T matrix with the nth row corresponding to
        T samples of the nth source in the kth dataset. This enforces the assumption of an equal
        number of samples in each dataset.
        For IVA, it is assumed that a source is statistically independent of all the other sources
        within the dataset and exactly dependent on at most one source in each of the other
        datasets.

    opt_approach : str, optional
        optimization type: 'gradient','newton','quasi'

    complex_valued : bool, optional
        if True, data pseudo cross-covariance matrices are calculated.
        If any input is complex, then complex_valued is forced to True

    circular : bool, optional
        set to True to only consider circular for complex-valued cost function

    whiten : bool, optional
        if True, data is whitened.
        For the complex-valued gradient and for quasi approach, whiten is forced to True.
        If data is not zero-mean, whiten is forced to True

    verbose : bool, optional
        enables print statements if True

    A : torch.tensor, optional
        true mixing matrices A of dimensions N x N x K, automatically sets verbose to True

    W_init : torch.tensor, optional
        initial estimate for demixing matrices in W, with dimensions N x N x K

    jdiag_initW : bool, optional
        if True, use CCA (K=2) / joint diagonalization (K>2) for initialization, else random

    max_iter : int, optional
        max number of iterations

    W_diff_stop : float, optional
        stopping criterion

    alpha0 : float, optional
        initial step size scaling (will be doubled for complex-valued gradient)

    return_W_change : bool, optional
        if the change in W in each iteration should be returned


    Returns
    -------
    W : torch.tensor
        the estimated demixing matrices of dimensions N x N x K so that ideally
        W[k] @ A[k] = P @ D[k], where P is any arbitrary permutation matrix and D[k] is any
        diagonal invertible (scaling) matrix. Note that P is common to all datasets. This is
        to indicate that the local permutation ambiguity between dependent sources
        across datasets should ideally be resolved by IVA.

    cost : float
        the cost for each iteration

    Sigma_N : torch.tensor
        Covariance matrix of each source component vector, with dimensions K x K x N

    isi : float
        joint isi (inter-symbol-interference) for each iteration. Only available if user
        supplies true mixing matrices for computing a performance metric. Else returns torch.nan.


    Notes
    -----

    Coded by Matthew Anderson (matt.anderson@umbc.edu)
    Converted to Python by Isabell Lehmann (isabell.lehmann@sst.upb.de)
    Converted to Pytorch by Gaspard Blaise and Clément Cosserat (clement.cosserat@inria.fr)

    References:

    [1] M. Anderson, X.-L. Li, & T. Adalı, "Nonorthogonal Independent Vector Analysis Using
    Multivariate Gaussian Model," LNCS: Independent Component Analysis and Blind Signal Separation,
    Latent Variable Analysis and Signal Separation, Springer Berlin / Heidelberg, 2010, 6365,
    354-361
    [2] M. Anderson, T. Adalı, & X.-L. Li, "Joint Blind Source Separation of Multivariate Gaussian
    Sources: Algorithms and Performance Analysis," IEEE Trans. Signal Process., 2012, 60, 1672-1683
    [3] M. Anderson, X.-L. Li, & T. Adalı, "Complex-valued Independent Vector Analysis: Application
    to Multivariate Gaussian Model," Signal Process., 2012, 1821-1831

    Version 01 - 20120913 - Initial publication
    Version 02 - 20120919 - Using decouple_trick function, bss_isi, whiten_data, & cca subfunctions
    Version 03 - 20210129 - Python version of the code, with max_iter=1024.

    """
    start = time.time()
    times = [0]

    if X.ndim != 3:
        raise AssertionError('X must have dimensions N x T x K.')
    elif X.shape[2] == 1:
        raise AssertionError('There must be ast least K=2 datasets.')

    if opt_approach != 'newton' and opt_approach != 'gradient' and opt_approach != 'quasi':
        raise AssertionError("opt_approach must be 'newton', 'gradient' or 'quasi'")

    # get dimensions
    N, T, K = X.shape

    if A is not None:
        supply_A = True
        A = torch.tensor(A) #Conversion of A in a Pytorch tensor if needed
        if A.shape[0] != N or A.shape[1] != N or A.shape[2] != K:
            raise AssertionError('A must have dimensions N x N x K.')
    else:
        supply_A = False

    blowup = 1e3
    # set alpha0 to max(alpha_min, alpha0*alpha_scale) when cost does not decrease
    alpha_scale = torch.tensor(0.9)
    alpha_min = torch.tensor(W_diff_stop)

    # if any input is complex, then 'complex_valued' must be True
    # complex_valued = complex_valued or torch.any(torch.is_complex(A)) or torch.any(torch.is_complex(X))

    # whitening is required for the quasi & complex-valued gradient approach
    whiten = whiten or (opt_approach == 'quasi') or (
            complex_valued and opt_approach == 'gradient')

    # test if data is zero-mean (test added by Isabell Lehmann)
    if torch.linalg.norm(torch.mean(X, dim=1)) > 1e-12:
            whiten = True

    if whiten:
        X, V = whiten_data_torch(X)

    # calculate cross-covariance matrices of X
    # R_xx = torch.einsum('nvk,mvl -> nmkl',X,X)
    
    R_xx = torch.zeros((N, N, K, K), dtype=X.dtype)
    for k1 in range(K):
        for k2 in range(k1, K):
            R_xx[:, :, k1, k2] = 1 / T * X[:, :, k1] @ torch.conj(X[:, :, k2].T)
            R_xx[:, :, k2, k1] = torch.conj(R_xx[:, :, k1, k2].T)  # R_xx is Hermitian

    # Check rank of data-covariance matrix: should be full rank, if not we inflate (this is ad hoc)
    # concatenate all covariance matrices in a big matrix
# Calculate R_xx_all and its rank
    R_xx_all = reshape_fortran(R_xx.permute(0, 2, 1, 3),(N * K, N * K))
    rank = torch.linalg.matrix_rank(R_xx_all)

    # Inflate Rx if necessary
    if rank < (N * K):
        _, k, _ = torch.linalg.svd(R_xx_all)
        R_xx_all += k[rank - 1] * torch.eye(N * K)  # add smallest singular value to main diagonal
        R_xx = R_xx_all.reshape(N, K, N, K, order='F').permute(0, 2, 1, 3)

    # Compute data pseudo cross-covariance matrices for complex-valued non-circular cost
    if complex_valued and not circular:
        # Initialize P_xx tensor
        P_xx = torch.zeros(N, N, K, K, dtype=torch.complex64)

        # Compute data pseudo cross-covariance matrices
        for k1 in range(K):
            for k2 in range(k1, K):
                P_xx[:, :, k1, k2] = 1 / T * torch.matmul(X[:, :, k1], X[:, :, k2].transpose(0, 1))
                P_xx[:, :, k2, k1] = torch.conj(P_xx[:, :, k1, k2].transpose(0, 1))  # P_xx is symmetric

    # Initializations

    # Initialize W
    if W_init is not None:
        if W_init.shape[0] != N or W_init.shape[1] != N or (
                W_init.ndim == 3 and W_init.shape[2] != K):
            raise AssertionError('W_init must have dimension N x N or N x N x K.')

        W = W_init.clone()

        # if not complex_valued and torch.any(torch.is_complex(W)) and opt_approach == 'gradient':
            # raise Warning('Supplied initial W is complex-valued and gradient option_approach is '
                        #   'set. Whitening option should be set to True for best performance.')
        if W.shape[0] == N and W.shape[1] == N and W.ndim == 2:
            W = torch.tile(W[:, :, torch.newaxis], [1, 1, K])

        if whiten:
            # V.permute(2,1,0)
            # W.permute(2,1,0)
            # W = torch.linalg.solve(V,W)
            # W.permute(2,1,0)
            for k in range(K):
                W[:, :, k] = torch.linalg.solve(V[:, :, k].T, W[:, :, k].T).T

    else:
        if jdiag_initW:
            if K > 2:
                # initialize with multi-set diagonalization (orthogonal solution)
                W = _jbss_sos(X, 0, 'whole')

            else:
                W = _cca(X)

        else:
            # randomly initialize
            W = torch.random.randn(N, N, K)
            if complex_valued:
                W = W + 1j * torch.randn(N, N, K)
            for k in range(K):
                W[:, :, k] = torch.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].T), W[:, :, k])

    if supply_A:
        isi = torch.zeros(max_iter)
        if whiten:
            # A matrix is conditioned by V if data is whitened
            A_w = torch.copy(A)
            A_w = torch.einsum('Nnk,nmk->Nmk',V,A_w)
            # for k in range(K):
            #     A_w[:, :, k] = V[:, :, k] @ A_w[:, :, k]
        else:
            A_w = torch.copy(A)
    else:
        isi = None

    # Initialize some local variables
    cost = torch.zeros(max_iter)
    cost_const = K * torch.log(2 * torch.pi * torch.exp(torch.tensor(1)))  # local constant

    # Initializations based on real vs complex-valued
    if complex_valued:
        grad = torch.zeros((N, K), dtype=torch.float64)
        if opt_approach == 'newton':
            HA = torch.zeros((N * K, N * K), dtype=complex)
        elif opt_approach == 'gradient':
            # Double step-size for complex-valued gradient optimization
            alpha0 = 2 * alpha0
        H = torch.zeros((N * K, N * K), dtype=complex)
    else:  # real-valued
        grad = torch.zeros((N, K),dtype=X.dtype)
        if opt_approach == 'newton':
            H = torch.zeros((N * K, N * K),dtype=X.dtype)

    # to store the change in W in each iteration
    W_change = []
    t0 = time.time()
    # Main Iteration Loop
    for iteration in range(max_iter):
        term_criterion = torch.tensor(0)

        # Some additional computations of performance via ISI when true A is supplied
        if supply_A:
            avg_isi, joint_isi = _bss_isi(W, A_w)
            isi[iteration] = joint_isi

        W_old = W.clone()  # save current W as W_old
        cost[iteration] = 0
        for k in range(K):
            cost[iteration] -= torch.log(torch.abs(torch.linalg.det(W[:, :, k])))

        Q = 0
        R = 0

        # Loop over each SCV
        for n in range(N):
            Wn = torch.conj(W[n, :, :]).flatten() #il faudra potentiellement enlever le conj

            # Efficient version of Sigma_n = 1/T * Y_n @ torch.conj(Y_n.T) with Y_n = W_n @ X_n
            if complex_valued:
                Sigma_n = torch.eye(K, dtype=complex)
            else:
                Sigma_n = torch.eye(K)

            for k1 in range(K):
                for k2 in range(k1, K):
                    Sigma_n[k1, k2] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ torch.conj(W[n, :, k2])
                    Sigma_n[k2, k1] = torch.conj(Sigma_n[k1, k2])

            if complex_valued and not circular:
                # complex-valued non-circular cost
                # compute SCV pseudo cross-covariance matrices
                Sigma_P_n = torch.zeros((K, K), dtype=complex)  # pseudo = 1/T * Y_n @ Y_n.T
                for k1 in range(K):
                    for k2 in range(k1, K):
                        Sigma_P_n[k1, k2] = W[n, :, k1] @ P_xx[:, :, k1, k2] @ W[n, :, k2]
                        Sigma_P_n[k2, k1] = Sigma_P_n[k1, k2]

                Sigma_total = torch.vstack(
                    [torch.hstack([Sigma_n, Sigma_P_n]),
                     torch.hstack([torch.conj(Sigma_P_n.T), torch.conj(Sigma_n)])])
                cost[iteration] += 0.5 * torch.log(torch.abs(torch.linalg.det(Sigma_total)))
            else:
                cost[iteration] += 0.5 * (cost_const + torch.log(torch.abs(torch.linalg.det(Sigma_n))))

            if complex_valued and not circular:
                Sigma_inv = torch.linalg.inv(Sigma_total)
                P = Sigma_inv[0:K, 0:K]
                Pt = Sigma_inv[0:K, K:2 * K]
            else:
                Sigma_inv = torch.linalg.inv(Sigma_n)

            hnk, Q, R = _decouple_trick_torch(W, n, Q, R)
            # Loop over each dataset
            for k in range(K):
                # Analytic derivative of cost function with respect to vn
                # Code below is efficient implementation of computing the gradient, which is
                # independent of T
                grad[:,k] = - hnk[:, k]/(W[n, :,k] @ hnk[:,k])

                if complex_valued and not circular:
                    for kk in range(K):
                        grad[:, k] += R_xx[:, :, k, kk] @ torch.conj(W[n, :, kk]) * torch.conj(P[k, kk]) \
                                      + P_xx[:, :, k, kk] @ W[n, :, kk] * torch.conj(Pt[k, kk])
                else:
                    for kk in range(K):
                        grad[:, k] += R_xx[:, :, k, kk] @ torch.conj(W[n, :, kk]) * Sigma_inv[kk, k]

                if opt_approach == 'gradient' or (opt_approach == 'quasi' and not complex_valued):
                    wnk = torch.conj(W[n, :, k])
                    if opt_approach == 'gradient':
                        grad_norm = _normalize_column_vectors(grad[:, k])
                        grad_norm_proj = _normalize_column_vectors(grad_norm - torch.conj(
                            wnk) @ grad_norm * wnk)  # non-colinear direction normalized
                        W[n, :, k] = torch.conj(
                            _normalize_column_vectors(wnk - alpha0 * grad_norm_proj))

                        for kk in range(K):  # = 1/T * Y_n @ torch.conj(Yn.T)
                            Sigma_n[k, kk] = W[n, :, k] @ R_xx[:, :, k, kk] @ torch.conj(W[n, :, kk])
                        Sigma_n[:, k] = torch.conj(Sigma_n[k, :])

                        if complex_valued and not circular:
                            for kk in range(K):  # = 1/T * Y_n @ * Y_n.T
                                Sigma_P_n[k, kk] = W[n, :, k] @ P_xx[:, :, k, kk] @ W[n, :, kk].mT
                            Sigma_P_n[:, k] = Sigma_P_n[k, :].T
                            Sigma_total = torch.vstack(
                                [torch.hstack([Sigma_n, Sigma_P_n]),
                                 torch.hstack([torch.conj(Sigma_P_n.mT), torch.conj(Sigma_n)])])
                            Sigma_inv = torch.linalg.inv(Sigma_total)
                        else:
                            Sigma_inv = torch.linalg.inv(Sigma_n)

                    else:  # real-valued quasi (block newton)
                        # Hessian inverse computed using matrix inversion lemma
                        H_inv = 1 / Sigma_inv[k, k] * (
                                torch.eye(N) - torch.outer(hnk[:, k], hnk[:, k]) / (
                                Sigma_inv[k, k] + 1 / (hnk[:, k] @ wnk) ** 2))

                        # Block-Newton update of Wnk
                        W[n, :, k] = _normalize_column_vectors(wnk - alpha0 * (H_inv @ grad[:, k]))

            if opt_approach == 'newton' or (opt_approach == 'quasi' and complex_valued):
                # Compute SCV Hessian
                for k1 in range(K):
                    if complex_valued:
                        if opt_approach == 'newton':  # complex-valued Newton
                            HA[k1 * N:k1 * N + N,
                            k1 * N:k1 * N + N] = torch.outer(
                                torch.conj(hnk[:, k1]), torch.conj(hnk[:, k1])) / (
                                                     torch.conj(hnk[:, k1] @ W[n, :, k1])) ** 2

                            if not circular:  # complex-valued Newton non-circular
                                HA[k1 * N:k1 * N + N, k1 * N:k1 * N + N] += Pt[k1, k1] * torch.conj(
                                    P_xx[:, :, k1, k1])
                        H[k1 * N:(k1 + 1) * N, k1 * N:(k1 + 1) * N] = torch.conj(
                            Sigma_inv[k1, k1] * R_xx[:, :, k1, k1])
                    else:  # real-valued Newton
                        H[k1 * N:k1 * N + N, k1 * N:k1 * N + N] = \
                            Sigma_inv[k1, k1] * R_xx[:, :, k1, k1] + torch.outer(
                                hnk[:, k1], hnk[:, k1]) / (hnk[:, k1] @ W[n, :, k1]) ** 2

                    for k2 in range(k1 + 1, K):
                        if opt_approach == 'newton' and complex_valued and not circular:
                            # complex-valued Newton non-circular
                            HA[k1 * N: k1 * N + N, k2 * N: k2 * N + N] = Pt[k2, k1] * torch.conj(
                                P_xx[:, :, k1, k2])
                            HA[k2 * N: k2 * N + N, k1 * N: k1 * N + N] = Pt[k1, k2] * torch.conj(
                                P_xx[:, :, k2, k1])
                            Hs = torch.conj(P[k2, k1] * R_xx[:, :, k1, k2])
                        else:
                            Hs = Sigma_inv[k1, k2] * R_xx[:, :, k2, k1].mT
                        H[k1 * N: k1 * N + N, k2 * N: k2 * N + N] = Hs
                        H[k2 * N: k2 * N + N, k1 * N: k1 * N + N] = torch.conj(Hs.mT)

                # Newton Update
                if not complex_valued:  # real-valued Newton
                    Wn -= alpha0 * torch.linalg.solve(H, grad.flatten()) #il faudra peut-être transposer avant de flatten
                else:  # complex-valued
                    if opt_approach == 'newton':  # complex-valued Newton
                        # improve stability by using 0.99 in Hessian
                        Wn -= alpha0 * torch.linalg.solve(
                            torch.conj(H) - 0.99 * torch.conj(HA) @ torch.linalg.solve(H, HA),
                            grad.flatten('F') - torch.conj(HA) @ torch.linalg.solve(H, torch.conj(
                                grad.flatten('F'))))
                    elif opt_approach == 'quasi':  # complex-valued Quasi-Newton
                        Wn -= alpha0 * torch.linalg.solve(torch.conj(H), grad.flatten('F'))
                    else:
                        raise AssertionError('Should not happen.')
                    Wn = torch.conj(Wn)

                # Store Updated W
                Wn = torch.reshape(Wn, (N, K)) # il faudra peut-être transposer avant le reshape
                for k in range(K):
                    W[n, :, k] = _normalize_column_vectors(Wn[:, k])

        for k in range(K):
            if complex_valued:  # for complex data, W_old @ W.T is not bounded between -1 and 1
                term_criterion = torch.maximum(term_criterion, torch.amax(torch.abs(
                    1 - torch.abs(torch.diag(W_old[:, :, k] @ torch.conj(W[:, :, k].mT))))))
            else:  # in original matlab code, this is the term criterion
                # (the result is the same as above for real data)
                term_criterion = torch.maximum(term_criterion, torch.amax(
                    1 - torch.abs(torch.diag(W_old[:, :, k] @ torch.conj(W[:, :, k].mT)))))

        W_change.append(term_criterion)

        # Decrease step size alpha if cost increased from last iteration
        if iteration > 0 and cost[iteration] > cost[iteration - 1]:
            alpha0 = torch.maximum(alpha_min, alpha_scale * alpha0)

        # Check the termination condition
        if term_criterion < W_diff_stop:
            break
        elif term_criterion > blowup or torch.isnan(cost[iteration]):
            for k in range(K):
                W[:, :, k] = torch.eye(N) + 0.1 * torch.randn(N, N)
            if complex_valued:
                W = W + 0.1j * torch.randn(N, N, K)
            if verbose:
                print('W blowup, restart with new initial value.')

        # Display Iteration Information
        if verbose:
            if supply_A:
                print(
                    f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                    f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
            else:
                print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')
    
    times.append(time.time()-t0)
    # Finish Display
    if iteration == 0 and verbose:
        if supply_A:
            print(
                f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
        else:
            print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Clean-up Outputs
    cost = cost[0:iteration + 1]

    if supply_A:
        isi = isi[0:iteration + 1]

    if whiten:
        for k in range(K):
            W[:, :, k] = W[:, :, k] @ V[:, :, k]
    else:  # no prewhitening
        # Scale demixing vectors to generate unit variance sources
        for n in range(N):
            for k in range(K):
                W[n, :, k] /= torch.sqrt(W[n, :, k] @ R_xx[:, :, k, k] @ torch.conj(W[n, :, k]))

    # Resort order of SCVs: Order the components from most to least ill-conditioned
    if not complex_valued or circular:
        P_xx = None
    if not whiten:
        V = None
    W, Sigma_N = _resort_scvs_torch(W, R_xx, whiten, V, complex_valued, circular, P_xx)

    end = time.time()

    if verbose:
        print(f"IVA-G finished after {(end - start) / 60} minutes with {iteration} iterations.")

    if return_W_change:
        return W, cost, Sigma_N, isi, W_change, times
    else:
        return W, cost, Sigma_N, isi, times


def fast_iva_g_torch(X, opt_approach='newton', complex_valued=False, circular=False, whiten=True,
          verbose=False, A=None, W_init=None, jdiag_initW=False, max_iter=1024,
          W_diff_stop=1e-6, alpha0=torch.tensor(1.0), return_W_change=False):
  
    start = time.time()
    times = [0]

    # get dimensions
    N, T, K = X.shape

    if A is not None:
        supply_A = True
        A = torch.tensor(A) #Conversion of A in a Pytorch tensor if needed
    else:
        supply_A = False

    blowup = 1e3
    # set alpha0 to max(alpha_min, alpha0*alpha_scale) when cost does not decrease
    alpha_scale = torch.tensor(0.9)
    alpha_min = torch.tensor(W_diff_stop)

    # whitening is required for the quasi & complex-valued gradient approach
    whiten = whiten or (opt_approach == 'quasi') or (opt_approach == 'gradient')

    # test if data is zero-mean (test added by Isabell Lehmann)
    if torch.linalg.norm(torch.mean(X, dim=1)) > 1e-12:
            whiten = True

    if whiten:
        X, V = fast_whiten_data_torch(X)
        V = V.permute(2,0,1)

    # calculate cross-covariance matrices of X
    R_xx = torch.einsum('nvk,mvl -> nmkl',X,X)/T
    R_xx = (R_xx + R_xx.permute(1, 0, 3, 2)) / 2

    # Check rank of data-covariance matrix: should be full rank, if not we inflate (this is ad hoc)
    # concatenate all covariance matrices in a big matrix
# Calculate R_xx_all and its rank
    R_xx_all = reshape_fortran(R_xx.permute(0, 2, 1, 3),(N * K, N * K))
    rank = torch.linalg.matrix_rank(R_xx_all)

    # Inflate Rx if necessary
    if rank < (N * K):
        _, k, _ = torch.linalg.svd(R_xx_all)
        R_xx_all += k[rank - 1] * torch.eye(N * K)  # add smallest singular value to main diagonal
        R_xx = R_xx_all.reshape(N, K, N, K, order='F').permute(0, 2, 1, 3)

    # Initializations

    # Initialize W
    if W_init is not None:
        W = W_init.clone()
        W = W.permute(2,0,1)
        if whiten:            
            W = torch.linalg.solve(V,W)

    else:
        # randomly initialize
        W = np.random.randn(K, N, N)
        W = np.linalg.solve(sc.linalg.sqrtm(np.einsum('knm,kNm->knN',W,W)), W)
        W = torch.Tensor(W)
        # W = torch.random.randn(N, N, K)
        # for k in range(K):
        #     W[:, :, k] = torch.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].mT), W[:, :, k])
        # W = W.permute(2,0,1)

    #Initialize Sigma
    Sigma = torch.einsum('knm,mMkK,KnM->nkK',W,R_xx,W)
    Sigma = (Sigma + Sigma.mT)/2
    Sigma_inv = torch.linalg.inv(Sigma)
        
    if supply_A:
        isi = torch.zeros(max_iter)
        if whiten:
            # A matrix is conditioned by V if data is whitened
            A_w = torch.copy(A)
            A_w = torch.einsum('kNn,nmk->Nmk',V,A_w)
        else:
            A_w = torch.copy(A)
    else:
        isi = None

    # Initialize some local variables
    cost = torch.zeros(max_iter)
    cost_const = K * torch.log(2 * torch.pi * torch.exp(torch.tensor(1)))  # local constant

    grad = torch.zeros((N, K),dtype=X.dtype)
    if opt_approach == 'newton':
        H = torch.zeros((N,N,K,K),dtype=X.dtype)

    # to store the change in W in each iteration
    W_change = []
    t0 = time.time()
    # Main Iteration Loop
    for iteration in range(max_iter):
        term_criterion = torch.tensor(0)

        # Some additional computations of performance via ISI when true A is supplied
        if supply_A:
            avg_isi, joint_isi = _bss_isi(W.permute(1,2,0), A_w)
            isi[iteration] = joint_isi

        W_old = W.clone()  # save current W as W_old
        cost[iteration] = 0
        det_W = torch.linalg.det(W)
        cost[iteration] -= torch.sum(torch.log(torch.abs(det_W)))

        Q = 0
        R = 0
           
        cost[iteration] += 0.5 * (cost_const + torch.sum(torch.log(torch.abs(torch.linalg.det(Sigma)))))

        # Loop over each SCV
        for n in range(N):
            hnk, Q, R = _decouple_trick_torch(W.permute(1,2,0), n, Q, R)
            # Loop over each dataset
            grad = - hnk/torch.einsum('kN,Nk->k',W[:,n,:],hnk)
            grad += torch.einsum('NMkK,KM,Kk->Nk',R_xx,W[:,n,:],Sigma_inv[n,:,:])
            
            if opt_approach == 'gradient' or (opt_approach == 'quasi'):
                
                if opt_approach == 'gradient':
                    grad_norm = grad/grad.norm(dim=0)  
                    grad_norm_proj = _normalize_column_vectors(grad_norm - torch.einsum('kN,Nk->k',W[:,n,:],grad_norm) * W[:,n,:].T) 
                    W[:, n, :] = _normalize_column_vectors(W[:,n,:].mT - alpha0 * grad_norm_proj).mT
                    
                else:  # real-valued quasi (block newton)
                    # Hessian inverse computed using matrix inversion lemma
                    aux1 = torch.diag(Sigma_inv[n,:,:]) + torch.einsum('kM,kM->k',hnk,W[:,n,:])**2
                    aux2 = torch.zeros(N,N,K) - torch.einsum('Nk,Mk->NMk',hnk,hnk)
                    H_inv = aux2 / torch.diag(Sigma_inv[n,:,:])*aux1 #H_inv is of dimension NxNxK

                    # Block-Newton update of Wnk
                    W[:, n, :] = _normalize_column_vectors(W[:, n, :].mT - alpha0 * (torch.einsum('nNk,Nk->nk',H_inv,grad))).mT

            if opt_approach == 'newton':
                # Compute SCV Hessian
                H = torch.einsum('Kk,NMkK->MNkK',Sigma_inv[n,:,:],R_xx)
                H[:,:,torch.arange(K),torch.arange(K)] += torch.einsum('nk,Nk->nNk',hnk,hnk)/torch.einsum('kn,nk->k',W[:,n,:],hnk)**2
                H = H.permute(0,2,1,3).reshape(K*N,K*N)
                H = (H + H.mT)/2

                # Newton Update
                W[:,n,:] -= alpha0 * torch.linalg.solve(H, (grad.mT).flatten()).reshape(K,N) 
                W[:,n,:] = _normalize_column_vectors(W[:,n,:].mT).mT
        
        #Initialize Sigma
        Sigma = torch.einsum('knm,mMkK,KnM->nkK',W,R_xx,W)
        Sigma = (Sigma + Sigma.mT)/2
        Sigma_inv = torch.linalg.inv(Sigma)
                        
        #check criterion
        term_criterion = torch.amax(1 - torch.abs(torch.sum(W*W_old,dim=2)))
        W_change.append(term_criterion)

        # Decrease step size alpha if cost increased from last iteration
        if iteration > 0 and cost[iteration] > cost[iteration - 1]:
            alpha0 = torch.maximum(alpha_min, alpha_scale * alpha0)

        # Check the termination condition
        if term_criterion < W_diff_stop:
            break
        elif term_criterion > blowup or torch.isnan(cost[iteration]):
            W = torch.eye(N).unsqueeze(2).repeat(1, 1, K) + 0.1 * torch.randn(N, N, K)
            W = W.permute(2,0,1)
    
    times.append(time.time()-t0)

    # Clean-up Outputs
    cost = cost[0:iteration + 1]

    if supply_A:
        isi = isi[0:iteration + 1]

    if whiten:
        W = torch.einsum('knN,kNM->knM',W,V)
    else:  # no prewhitening
        # Scale demixing vectors to generate unit variance sources
        W /= torch.sqrt(torch.einsum('kNn,nmk,kMm->kNM',W,R_xx[:,:,torch.arange(K),torch.arange(K)],W))
        # for n in range(N):
        #     for k in range(K):
        #         W[n, :, k] /= torch.sqrt(W[n, :, k] @ R_xx[:, :, k, k] @ torch.conj(W[n, :, k]))

    # Resort order of SCVs: Order the components from most to least ill-conditioned
    if not whiten:
        V = None
    W = W.permute(1,2,0)
    V = V.permute(1,2,0)
    W, Sigma_N = _resort_scvs_torch(W, R_xx, whiten, V, complex_valued, circular, None)

    if return_W_change:
        return W, cost, Sigma_N, isi, W_change, times
    else:
        return W, cost, Sigma_N, isi, times


