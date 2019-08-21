"""network.py

Author: @omarschall, 8-20-2019"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multilayer_Perceptron(nn.Module):

    def __init__(self, n_in, n_out, N_h,
                 activation=F.relu, output=F.log_softmax):

        super(Multilayer_Perceptron, self).__init__()
        self.N_h = N_h
        self.n_in = n_in
        self.n_out = n_out
        self.layer_sizes = [n_in] + N_h + [n_out]
        self.n_layers = len(self.layer_sizes)
        self.activation = activation
        self.output = output
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            setattr(self, attr, nn.Linear(self.layer_sizes[i-1],
                                          self.layer_sizes[i]))

    def forward(self, x):
        x = x.view(-1, self.layer_sizes[0])
        for i_layer in range(1, self.n_layers):
            attr = 'layer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < self.n_layers - 1:
                x = self.activation(x)

        return self.output(x, dim=1)
