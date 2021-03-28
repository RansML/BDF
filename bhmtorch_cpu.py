"""
# 3D Bayesian Dynamic Fields with pytorch
# Ransalu Senanayake, Jason Zheng, and Kyle Hatch
"""
import math
import numpy as np
import torch
import statsmodels.api as sm

from kernel import rbf_kernel_conv, rbf_kernel_wasserstein, rbf_kernel

class BHM_VELOCITY_PYTORCH:
    def __init__(self, alpha=None, beta=None, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, kernel_type='rbf', likelihood_type="gamma", device='cpu', w_hatx=None, w_haty=None, w_hatz=None):
        self.nIter = nIter
        self.rbf_kernel_type = kernel_type
        self.likelihood_type = likelihood_type

        if device == 'cpu':
            self.device = torch.device("cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")

        self.alpha = alpha
        self.beta = beta

        self.gamma = torch.tensor(gamma, device=self.device)
        if self.gamma.shape[0] > 2:
            self.gamma = self.gamma[:2]

        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)

        if w_hatx is not None:
            self.w_hatx = w_hatx

        if w_haty is not None:
            self.w_haty = w_haty

        if w_hatz is not None:
            self.w_hatz = w_hatz

    def updateMuSig(self, mu_x, sig_x, mu_y, sig_y, mu_z, sig_z):
        """
        :param mu: mean
        :param sig: variance
        """
        self.mu_x = mu_x
        self.sig_x = sig_x

        self.mu_y = mu_y
        self.sig_y = sig_y

        self.mu_z = mu_z
        self.sig_z = sig_z

    def fit(self, X, y_vx, y_vy, y_vz, eps=1e-10):
        """
        :param X: raw data
        :param y: labels
        """
        if self.likelihood_type == "gamma":
            return self.fit_gamma_likelihood(X, y_vx, y_vy, y_vz, eps)
        elif self.likelihood_type == "gaussian":
            return self.fit_gaussian_likelihood(X, y_vx, y_vy, y_vz)
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(self.likelihood_type))


    def fit_gamma_likelihood(self, X, y_vx, y_vy, y_vz, eps=1e-10):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X, self.rbf_kernel_type)

        all_ys = torch.cat((y_vx, y_vy, y_vz), dim=-1)

        X = X.double()
        y_vx = y_vx.double()
        y_vy = y_vy.double()
        y_vz = y_vz.double()

        y_vx = torch.log(y_vx + eps)
        y_vy = torch.log(y_vy + eps)
        y_vz = torch.log(y_vz + eps)
        lam = 0.1

        self.w_hatx = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vx).double())
        self.w_haty = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vy).double())
        self.w_hatz = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vz).double())


    def fit_gaussian_likelihood(self, X, y_vx, y_vy, y_vz):
        """
        :param X: raw data
        :param y: labels
        """
        print(" Data shape:", X.shape)
        X = self.__sparse_features(X, self.rbf_kernel_type)
        print(" Kernelized data shape:", X.shape)
        print(" Hinge point shape:", self.grid.shape)

        self.mu_x, self.sig_x = self.__calc_posterior(X, y_vx)
        self.mu_y, self.sig_y = self.__calc_posterior(X, y_vy)
        self.mu_z, self.sig_z = self.__calc_posterior(X, y_vz)
        return self.mu_x, self.sig_x, self.mu_y, self.sig_y, self.mu_z, self.sig_z

    def predict(self, Xq, query_blocks=-1, variance_only=False):
        """
        :param Xq: raw inquery points
        :return: mean velocity
        """
        if self.likelihood_type == "gamma":
            return self.predict_gamma_likelihood(Xq, query_blocks, variance_only)
        elif self.likelihood_type == "gaussian":
            return self.predict_gaussian_likelihood(Xq, query_blocks, variance_only)
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(self.likelihood_type))

    def predict_gaussian_likelihood(self, Xq, query_blocks=-1, variance_only=False):
        """
        :param Xq: raw inquery points
        :return: mean and variance of velocity
        """

        Nq, M = Xq.shape[0], self.grid.shape[0]

        Xq = Xq.float()
        print(" Query data shape:", Xq.shape)

        if query_blocks <= 0:
            Xq = self.__sparse_features(Xq, self.rbf_kernel_type)  # .double()
            print(" Kernelized query data shape:", Xq.shape)

            if variance_only:
                sig2_inv_a_x = 1/self.beta + diag_only_mm(Xq.mm(self.sig_x), Xq.t()) # (635, 2508) X (2508, 2508) --> (635, 2508), (635, 2508) X (2508, 635) --> (635, 635)
                sig2_inv_a_y = 1/self.beta + diag_only_mm(Xq.mm(self.sig_y), Xq.t())
                sig2_inv_a_z = 1/self.beta + diag_only_mm(Xq.mm(self.sig_z), Xq.t())
            else:
                sig2_inv_a_x = 1/self.beta + Xq.mm(self.sig_x).mm(Xq.t()) # (635, 2508) X (2508, 2508) --> (635, 2508), (635, 2508) X (2508, 625) --> (635, 635)
                sig2_inv_a_y = 1/self.beta + Xq.mm(self.sig_y).mm(Xq.t())
                sig2_inv_a_z = 1/self.beta + Xq.mm(self.sig_z).mm(Xq.t())
        else:
            step_size = Xq.shape[0] // query_blocks
            if Nq % step_size != 0:
                query_blocks += 1

            mu_a_x = torch.zeros((Nq, 1))
            mu_a_y = torch.zeros((Nq, 1))
            mu_a_z = torch.zeros((Nq, 1))

            if variance_only:
                sig2_inv_a_x = torch.zeros((Nq,))
                sig2_inv_a_y = torch.zeros((Nq,))
                sig2_inv_a_z = torch.zeros((Nq,))
            else:
                sig2_inv_a_x = torch.zeros((Nq, Nq))
                sig2_inv_a_y = torch.zeros((Nq, Nq))
                sig2_inv_a_z = torch.zeros((Nq, Nq))

            for i in range(query_blocks):
                start = i * step_size
                end = start + step_size
                if end > Nq:
                    end = Nq

                Xq_block_i = self.__sparse_features(Xq[start:end], self.rbf_kernel_type)  # .double()
                print(" Kernelized query data shape {} in block {} out of {}".format(Xq_block_i.shape, i, query_blocks))

                mu_a_x[start:end] = Xq_block_i.mm(self.mu_x.reshape(-1, 1))#.squeeze()
                mu_a_y[start:end] = Xq_block_i.mm(self.mu_y.reshape(-1, 1))#.squeeze()
                mu_a_z[start:end] = Xq_block_i.mm(self.mu_z.reshape(-1, 1))#.squeeze()

                if variance_only:
                    #print("VARIANCE ONLY")
                    sig2_inv_a_x[start:end] = 1/self.beta + diag_only_mm(Xq_block_i.mm(self.sig_x), Xq_block_i.t())
                    sig2_inv_a_y[start:end] = 1/self.beta + diag_only_mm(Xq_block_i.mm(self.sig_y), Xq_block_i.t())
                    sig2_inv_a_z[start:end] = 1/self.beta + diag_only_mm(Xq_block_i.mm(self.sig_z), Xq_block_i.t())
                else:
                    #print("NO VARIANCE ONLY")
                    for j in range(query_blocks):
                        start2 = j * step_size
                        end2 = start2 + step_size
                        if end2 > Xq.shape[0]:
                            end2 = Xq.shape[0]

                        Xq_block_2 = self.__sparse_features(Xq[start2:end2], self.rbf_kernel_type)
                        sig2_inv_a_x[start:end, start2:end2] = 1/self.beta + Xq_block_i.mm(self.sig_x).mm(Xq_block_2.t())
                        sig2_inv_a_y[start:end, start2:end2] = 1/self.beta + Xq_block_i.mm(self.sig_y).mm(Xq_block_2.t())
                        sig2_inv_a_z[start:end, start2:end2] = 1/self.beta + Xq_block_i.mm(self.sig_z).mm(Xq_block_2.t())

        if variance_only:
            sig2_inv_a_x, sig2_inv_a_y, sig2_inv_a_z = sig2_inv_a_x.view(-1,1), sig2_inv_a_y.view(-1,1), sig2_inv_a_z.view(-1,1)

        return mu_a_x, sig2_inv_a_x, mu_a_y, sig2_inv_a_y, mu_a_z, sig2_inv_a_z

    def predict_gamma_likelihood(self, Xq, query_blocks=-1):
        """
        :param Xq: raw inquery points
        :return: mean velocity in each direction
        """
        print(" Query data shape:", Xq.shape)
        Xq = self.__sparse_features(Xq, self.rbf_kernel_type).double()
        print(" Kernelized query data shape:", Xq.shape)

        if query_blocks <= 0:
            mean_x, mean_y, mean_z = torch.exp(Xq.mm(self.w_hatx)), torch.exp(Xq.mm(self.w_haty)), torch.exp(Xq.mm(self.w_hatz))
        else:
            mean_x_, mean_y_, mean_z_ = torch.exp(Xq.mm(self.w_hatx)), torch.exp(Xq.mm(self.w_haty)), torch.exp(Xq.mm(self.w_hatz))

            step_size = Xq.shape[0] // query_blocks
            if Xq.shape[0] % step_size != 0:
                query_blocks += 1

            mu_a_x = torch.zeros((Xq.shape[0], 1))
            mu_a_y = torch.zeros((Xq.shape[0], 1))
            mu_a_z = torch.zeros((Xq.shape[0], 1))
            sig2_inv_a_x = torch.zeros((Xq.shape[0], Xq.shape[0]))
            sig2_inv_a_y = torch.zeros((Xq.shape[0], Xq.shape[0]))
            sig2_inv_a_z = torch.zeros((Xq.shape[0], Xq.shape[0]))

            for i in range(query_blocks):
                start = i * step_size
                end = start + step_size
                if end > Xq.shape[0]:
                    end = Xq.shape[0]


        return mean_x, mean_y, mean_z

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = torch.meshgrid(torch.arange(x_min, x_max, cell_resolution[0]), \
                             torch.arange(y_min, y_max, cell_resolution[1]), \
                             torch.arange(z_min, z_max, cell_resolution[2]))

        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


    def __sparse_features(self, X, rbf_kernel_type='conv'):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if rbf_kernel_type == 'conv':
            raise NotImplementedError
            # rbf_features = rbf_kernel_conv(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        elif rbf_kernel_type == 'wass':
            raise NotImplementedError
            # rbf_features = rbf_kernel_wasserstein(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        else:
            rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return rbf_features

    def __calc_posterior(self, X, y):
        """
        :param X: input features
        :param y: labels
        :return: new_mean, new_varaiance
        """
        order = X.shape[1]
        theta = X.numpy()

        A = self.beta*theta.T.dot(theta) + self.alpha*np.eye((order))
        sig = np.linalg.pinv(A)
        mu = self.beta*sig.dot(theta.T.dot(y))

        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32) # change to double??


def diag_only_mm(x, y):
    return (x * y.T).sum(-1)
