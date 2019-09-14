"""network.py

Author: @omarschall, 8-20-2019"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pdb import set_trace
from copy import copy
import numpy as np
from utils import *

class Net(nn.Module):

    def __init__(self, layer_sizes, lr_init):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            setattr(self, attr, nn.Linear(layer_sizes[i - 1],
                                          layer_sizes[i]))
            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.A = np.random.normal(0, 1, self.n_params)
        self.B = np.random.normal(0, 1, self.n_params)
        self.eta = np.ones(self.n_params) * lr_init

    def forward(self, x):
        x = x.view(-1, self.layer_sizes[0])
        for i_layer in range(1, self.n_layers):
            attr = 'layer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < self.n_layers - 1:
                x = F.relu(x)

        return F.log_softmax(x, dim=1)

    def UORO_update_step(self, Q):

        self.nu = np.random.choice([-1, 1], Q.shape)
        grad = self.flatten_array([p.grad.data.numpy() for p in self.parameters()])

        self.M_projection = self.nu * grad
        self.A_forwards = self.A - self.eta * Q
        A_norm = norm(self.A_forwards)
        B_norm = norm(self.B)
        M_norm = norm(self.M_projection)
        self.rho_0 = np.sqrt(B_norm / A_norm)
        self.rho_1 = np.sqrt(M_norm / np.sqrt(len(self.nu)))

        self.A = self.rho_0 * self.A_forwards + self.rho_1 * self.nu
        self.B = (1 / self.rho_0) * self.B + (1 / self.rho_1) * self.M_projection

    def get_updated_eta(self, mlr, test_grad):

        test_grad = self.flatten_array(test_grad)
        self.eta -= mlr * (test_grad.dot(self.A)) * self.B
        self.eta = np.maximum(0, self.eta)

        return np.copy(self.unflatten_array(self.eta))

    def flatten_array(self, X):
        """Takes list of arrays in natural shape of the network parameters
        and returns as a flattened 1D numpy array."""

        return np.concatenate([x.flatten() for x in X])

    def unflatten_array(self, X):
        """Takes flattened array and returns in natural shape for network
        parameters."""

        N = self.param_cumsum

        return [np.reshape(X[N[i]:N[i + 1]], s) for i, s in enumerate(self.param_shapes)]

