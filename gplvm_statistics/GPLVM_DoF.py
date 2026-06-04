#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, svd
from typing import Optional, Union
from pyspoc import ReducedStatistic



class GPLVM_dof(ReducedStatistic):
    """
    Calculate the Effective Degrees of Freedom (DoF) of the GPLVM for a given dataset.
    (MAP Version )
    """

    def __init__(self, ARD_initial_dimension=15, max_iters=200):
        super().__init__()
        self.ARD_initial_dimension = ARD_initial_dimension
        self.max_iters = max_iters

    @property
    def name(self) -> str:
        return "GPLVM-DoF-Estimator"

    @property
    def identifier(self) -> str:
        return "gplvm_dof_estimator_numpy"

    @property
    def labels(self) -> list[str]:
        return ["manifold", "gplvm", "complexity", "dof"]

    # GPLVM

    def _kernel_rbf(self, X, alpha, gamma, beta):
        """calculate RBF kernel"""
        n = X.shape[0]
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = alpha * np.exp(-0.5 * gamma * sq_dists)
        K += np.eye(n) * (1.0 / beta) # white noise
        return K, sq_dists

    def _log_likelihood(self, params, Y, q, n, d, return_grad=True):
        """
        Calculate Objective.
        """
        # 1. unpack parameters
        X_flat = params[:-3]
        X = X_flat.reshape(n, q)
        
        # 
        log_alpha, log_gamma, log_beta = params[-3:]
        log_alpha = np.clip(log_alpha, -20, 25)
        log_gamma = np.clip(log_gamma, -20, 25)
        log_beta  = np.clip(log_beta,  -20, 25)
        
        alpha = np.exp(log_alpha)
        gamma = np.exp(log_gamma)
        beta = np.exp(log_beta)

        # 2. calculate RBF kernel
        K, sq_dists = self._kernel_rbf(X, alpha, gamma, beta)

        # 3. Cholesky Decomposition
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e100, np.zeros_like(params) if return_grad else 1e100

        # 4. Compute Log Likelihood P(Y|X)
        K_inv_Y = cho_solve((L, True), Y)
        data_fit = -0.5 * np.trace(np.dot(Y.T, K_inv_Y))
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        complexity = -0.5 * d * log_det_K
        constant = -0.5 * d * n * np.log(2 * np.pi)

        log_likelihood = complexity + data_fit + constant
        # log_prior
        log_prior = -0.5 * np.sum(X**2) - 0.5 * n * q * np.log(2 * np.pi)
        
        # objective
        NLL = -(log_likelihood + log_prior)

        if not return_grad:
            return NLL

        # 5. Gradient Computation
        K_inv = cho_solve((L, True), np.eye(n))
        dL_dK = 0.5 * (np.dot(K_inv_Y, K_inv_Y.T) - d * K_inv)
        dNLL_dK = -dL_dK 

        # Hyperparameters Gradients
        K_rbf = K - np.eye(n) * (1.0 / beta)
        d_alpha = np.sum(dNLL_dK * (K_rbf / alpha)) * alpha
        d_gamma = np.sum(dNLL_dK * (-0.5 * sq_dists * K_rbf)) * gamma
        d_beta = np.trace(dNLL_dK * (-1.0 / beta**2 * np.eye(n))) * beta

        # X Gradient
        G = dNLL_dK * K_rbf
        dLik_dX = np.zeros_like(X) 
        for d_idx in range(q):
            dist_matrix = np.subtract.outer(X[:, d_idx], X[:, d_idx])
            dLik_dX[:, d_idx] = -gamma * np.sum(G * dist_matrix, axis=1) * 2.0
        #     
        dNLL_dX = dLik_dX + X

        grads = np.concatenate([dNLL_dX.flatten(), [d_alpha, d_gamma, d_beta]])
        
        # 
        if np.isnan(grads).any() or np.isinf(grads).any():
            return 1e100, np.zeros_like(params)
            
        return NLL, grads

    def compute(self, data: np.ndarray) -> np.ndarray: 
        """
        Fit GPLVM and return ONLY the ESTIMATED DoF.
        """
        # pre
        Y = np.array(data, copy=True)
        if np.isnan(Y).any(): return np.array([0.0])
        
        # standardise 
        # 
        Y_std = np.std(Y)
        if Y_std < 1e-9: Y_std = 1.0
        Y = (Y - np.mean(Y, axis=0)) / Y_std
        
        n, d_out = Y.shape
        q = min(self.ARD_initial_dimension, d_out)

        # initialisation
        # PCA initialisation
        u, s, vt = svd(Y, full_matrices=False)
        X_init = u[:, :q] * s[:q]

        # initialise hyperparameters
        params_init = np.concatenate([
            X_init.flatten(), 
            [np.log(1.0), -np.log(q), np.log(10.0)] 
        ])

        
        
        res = minimize(
            fun=self._log_likelihood,
            x0=params_init,
            args=(Y, q, n, d_out, True),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.max_iters}
        )
        
        

        # calculate DoF
        optimized_params = res.x
        X_opt = optimized_params[:-3].reshape(n, q)
        log_alpha, log_gamma, log_beta = optimized_params[-3:]
        
        #
        log_alpha = np.clip(log_alpha, -20, 25)
        log_gamma = np.clip(log_gamma, -20, 25)
        log_beta  = np.clip(log_beta,  -20, 25)
        
        alpha = np.exp(log_alpha)
        gamma = np.exp(log_gamma)
        beta = np.exp(log_beta)

        # clean kernel
        K_clean, _ = self._kernel_rbf(X_opt, alpha, gamma, beta=np.inf)
        
        # e-val
        eigvals = np.linalg.eigvalsh(K_clean)
        eigvals = np.maximum(eigvals, 0)
        
        # DoF
        noise_var = 1.0 / beta
        dof = np.sum(eigvals / (eigvals + noise_var))
        
        
        # return DoF
        return np.array([dof])

