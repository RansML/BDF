import math
import numpy as np
import torch

from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel as sklearn_kernel

DEFAULT_DEVICE = torch.device("cpu")

# ==============================================================================
# Kernel Method Implementations using PyTorch
# ==============================================================================
def rbf_kernel_conv(X, Y, gamma, sigma, device=DEFAULT_DEVICE):
    """
    Vectorized implementation
    Performs rbf kernel convolution on input distributions and hinge point grid
    """
    N, d = X.shape
    if gamma is None:
        gamma = 1. / d
    if sigma is None:
        sigma = torch.zeros(N)

    if sigma.unsqueeze(-1).shape[1] == 1 and gamma.unsqueeze(-1).shape[0] == 1:
        K = torch.from_numpy(euclidean_distances(X, Y, squared=True)).float().to(device)
        sig_terms = 2*sigma**2
        return (1/((1+gamma*sig_terms)**(1/2))).unsqueeze(-1) * torch.exp(-1*(K/((1/gamma) + sig_terms).unsqueeze(-1)))

    # Handle Multi-dim sigma/gamma
    M = Y.shape[0]
    if len(sigma.shape) == 1:
        sigma = sigma.unsqueeze(-1)
    det = (torch.det(torch.diag_embed(2*gamma*sigma+1))**(-0.5)).unsqueeze(-1)
    diff = X.unsqueeze(1)-Y
    p = torch.diag_embed((2*sigma+1/gamma)**(-1))
    return det * torch.exp(-1 *((diff@p).view(N*M, 1, d)).bmm(diff.view(N*M, d, 1)).view(N,M))


def rbf_kernel_wasserstein(X, Y, gamma, sigma=None, device=DEFAULT_DEVICE):
    """
    Vectorized implementation
    Performs rbf kernel wasserstein on input distributions and hinge point grid
    """
    N, d = X.shape
    if gamma is None:
        gamma = 1. / d
    if sigma is None:
        sigma = torch.zeros(N)

    if gamma.unsqueeze(-1).shape[0] == 1:
        K = torch.from_numpy(euclidean_distances(X, Y, squared=True)).float().to(device)
        if sigma.unsqueeze(-1).shape[1] == 1:
            return torch.exp(-1*gamma*(K+(sigma**2).unsqueeze(-1)))
        covar_trace = torch.sum(sigma)
        return torch.exp(-1*gamma*(K+covar_trace))
    else:
        M = Y.shape[0]
        gamma = torch.diag(gamma)
        diff = X.unsqueeze(1)-Y
        covar_trace = torch.sum(sigma)
        return torch.exp(-1*((diff@gamma).view(N*M, 1, d)).bmm(diff.view(N*M, d, 1)).view(N,M) + covar_trace)


def rbf_kernel(X, Y, gamma, sigma=None, device=DEFAULT_DEVICE):
    """
    Vectorized Implementations
    Performs RBF kernel method using ARD with 2 and 3 dimensional gamma
    """
    if gamma.unsqueeze(-1).shape[0] == 1:
        return torch.tensor(sklearn_kernel(X.cpu().detach().numpy(), Y.cpu().detach().numpy(), gamma.cpu().detach().numpy()), device=device)

    else:
        print("In second")
        K = torch.from_numpy(euclidean_distances(X, Y, squared=True)).float().to(device)
        return torch.exp(-1/2 * torch.sum(gamma)*K)
