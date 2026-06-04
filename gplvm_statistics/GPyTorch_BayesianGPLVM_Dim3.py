#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyspoc import ReducedStatistic
import gpytorch


# GPyTorch Bayesian GPLVM
class _GPyTorch_BayesianGPLVM(gpytorch.models.ApproximateGP):
    def __init__(self, n, d_out, latent_dim, X_init, n_inducing=30):
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
        
        self.q_mu = nn.Parameter(X_init.clone())
        self.q_log_std = nn.Parameter(torch.ones(n, latent_dim) * -2.0)
        
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([d_out]))
        
        # using log-normal
        # more sensitive
        ls_prior = gpytorch.priors.LogNormalPrior(loc=2.0, scale=1.0)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=latent_dim, 
                batch_shape=torch.Size([d_out]),
                lengthscale_prior=ls_prior
            ),
            batch_shape=torch.Size([d_out])
        )

    def forward(self, X):
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


# Effective dimensions
class GPyTorch_BayesianGPLVM_Dim3(ReducedStatistic):
    def __init__(self, ARD_initial_dimension=15, max_iters=300, lr=0.05):
        super().__init__()
        self.ARD_initial_dimension = ARD_initial_dimension
        self.max_iters = max_iters
        self.lr = lr

    @property
    def name(self) -> str:
        return "GPLVM-GPyTorch-ActiveDims"

    @property
    def identifier(self) -> str:
        return "gplvm_gpytorch_activedims"

    @property
    def labels(self) -> list[str]:
        return ["manifold", "bayesian", "ard", "active_dimensions", "sparsity_prior"]

    def compute(self, data: np.ndarray) -> np.ndarray:
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
        u, s, v = torch.pca_lowrank(Y_tensor, q=q_init, center=False, niter=5)
        X_init = u * s

        # BGPLVM
        model = _GPyTorch_BayesianGPLVM(n, d_out, q_init, X_init, n_inducing=min(50, n))
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d_out)

        model.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
        
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=self.lr)

        # KL weight
        kl_weight = float(d_out) 

        for i in range(self.max_iters):
            optimizer.zero_grad()
            X_sample = model.sample_latent()
            output = model(X_sample)
            
            # loss
            loss_gp = -mll(output, Y_tensor) 
            # kl divergence
            loss_kl_x = (model.get_kl_x() * kl_weight) / n 
            
            loss = loss_gp + loss_kl_x
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            X_opt = model.q_mu.detach()
            latent_std = X_opt.std(dim=0)
            
            # Inverse Lengthscale
            ls = model.covar_module.base_kernel.lengthscale.squeeze(1).mean(dim=0)
            inv_ls = 1.0 / ls
            
            # calculate score
            raw_score = inv_ls * latent_std
            
            
            # model collapsed
            if latent_std.max() < 0.05:
                active_dims_count = self.ARD_initial_dimension
                contribution_ratio = torch.zeros_like(raw_score)
                
            else:
                sum_score = raw_score.sum()
                contribution_ratio = raw_score / sum_score
                
                # ordered contributions
                sorted_ratios, indices = torch.sort(contribution_ratio, descending=True)
                cumulative_ratios = torch.cumsum(sorted_ratios, dim=0)
                
                active_dims_count = 0
                for i in range(len(sorted_ratios)):
                    # contribution over 5%
                    if sorted_ratios[i] > 0.05:
                        active_dims_count += 1
                        
                    # cumulative contribution over 90%
                    if cumulative_ratios[i] >= 0.90:
                        break
            
            

        return np.array([float(active_dims_count)])

