#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union
from pyspoc import ReducedStatistic
import gpytorch




# GPyTorch Bayesian GPLVM
class _GPyTorch_BayesianGPLVM(gpytorch.models.ApproximateGP):
    def __init__(self, n, d_out, latent_dim, X_init, n_inducing=30):
        # 1. inducing variational
        inducing_points = torch.randn(d_out, n_inducing, latent_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            n_inducing, batch_shape=torch.Size([d_out])
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=d_out
        )
        super().__init__(variational_strategy)
        
        # 2. latet posterior q(X)
        self.q_mu = nn.Parameter(X_init.clone())
        self.q_log_std = nn.Parameter(torch.ones(n, latent_dim) * -2.0)
        
        # 3. GP kernel 
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([d_out]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim, batch_shape=torch.Size([d_out])),
            batch_shape=torch.Size([d_out])
        )

    def forward(self, X):
        # forward X
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def sample_latent(self):
        q_std = torch.exp(self.q_log_std)
        epsilon = torch.randn_like(self.q_mu)
        return self.q_mu + q_std * epsilon

    def get_kl_x(self):
        q_std = torch.exp(self.q_log_std)
        kl = -0.5 * torch.sum(1 + 2 * self.q_log_std - self.q_mu.pow(2) - q_std.pow(2))
        return kl


# Beta
class GPyTorch_BayesianGPLVM_Beta(ReducedStatistic):
    def __init__(self, ARD_initial_dimension=15, max_iters=200, lr=0.05):
        super().__init__()
        self.ARD_initial_dimension = ARD_initial_dimension
        self.max_iters = max_iters
        self.lr = lr

    @property
    def name(self) -> str:
        return "GPLVM-GPyTorch-Beta"

    @property
    def identifier(self) -> str:
        return "gplvm_gpytorch_beta"

    @property
    def labels(self) -> list[str]:
        return ["manifold", "bayesian", "gpytorch", "beta", "noise_estimation"]

    def compute(self, data: np.ndarray) -> np.ndarray:
        # 1. pre
        Y = np.array(data, copy=True)
        if np.isnan(Y).any(): return np.array([0.0])
        
        # standardise
        Y_std = np.std(Y, axis=0)
        Y_std[Y_std < 1e-9] = 1.0
        Y_centered = (Y - np.mean(Y, axis=0)) / Y_std
        
        n, d_out = Y_centered.shape
        q_init = min(self.ARD_initial_dimension, d_out)
        Y_tensor = torch.tensor(Y_centered, dtype=torch.float32)

        # PCA initialisation
        u, s, v = torch.pca_lowrank(Y_tensor, q=q_init, center=False, niter=3)
        X_init = u * s

        # 2. initialise model
        model = _GPyTorch_BayesianGPLVM(n, d_out, q_init, X_init, n_inducing=min(30, n))
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d_out)

        model.train()
        likelihood.train()

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
        
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=self.lr)

        # 3. train
        # print(f"DEBUG:{Y_centered.shape}")
        for i in range(self.max_iters):
            optimizer.zero_grad()
            X_sample = model.sample_latent()
            output = model(X_sample)
            
            loss_gp = -mll(output, Y_tensor) 
            loss_kl_x = model.get_kl_x() / n
            loss = loss_gp + loss_kl_x
            
            loss.backward()
            optimizer.step()

        # 4. extract Beta
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            # noise_variance
            # use .mean()
            noise_var = likelihood.noise.mean().item()
            
            
            beta_opt = 1.0 / noise_var
            
            # print(f"DEBUG: Real Noise Variance: {noise_var:.6f}")
            # print(f"DEBUG: Estimated Beta: {beta_opt:.4f}")

        return np.array([float(beta_opt)])

