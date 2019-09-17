"""simulation.py

Author: @omarschall, 9-13-2019"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from itertools import cycle
from utils import *
from copy import copy

class Simulation:

    def __init__(self, model, optimizer, monitors=[], Hess_est_r=0.001,
                 mlr=0.000001, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.monitors = monitors
        self.Hess_est_r = Hess_est_r
        self.mlr = mlr

        self.report_interval = 1000
        self.__dict__.update(kwargs)

    def run(self, meta_train_loader, meta_test_loader, val_loader, test_loader,
            mode='train', n_epochs=1):

        #Initialize monitors
        self.mons = {mon:[] for mon in self.monitors}

        self.mode = mode
        if self.mode == 'train':
            data_loader = meta_train_loader
        elif self.mode == 'test':
            data_loader = meta_test_loader
            
        self.val_loader = cycle(val_loader)
        self.test_loader = test_loader

        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                
                training_loss = self.train_step(batch_idx, data, target)
    
                if batch_idx % self.report_interval == 0 and batch_idx > 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader), training_loss.item()))
                    
                self.update_monitors()
            
            self.test()

        #Delete validation data to save space
        del(self.val_loader)
        del(self.test_loader)

    def train_step(self, batch_idx, data, target):

        self.optimizer.zero_grad()
        output = self.model(data)
        training_loss = F.nll_loss(output, target)
        training_loss.backward()
        self.optimizer.step()

        ### --- METALEARNING --- ###
        if self.mode == 'train':
    
            # Approximate the Hessian
            A = self.model.unflatten_array(self.model.A)
    
            model_plus = copy(self.model)
            for p, a in zip(model_plus.parameters(), A):
                perturbation = torch.from_numpy(self.Hess_est_r * a).type(torch.FloatTensor)
                p.data += perturbation
    
            model_plus.train()
            output = model_plus(data)
            loss = F.nll_loss(output, target)
            loss.backward()
    
            model_minus = copy(self.model)
            for p, a in zip(model_minus.parameters(), A):
                perturbation = torch.from_numpy(self.Hess_est_r * a).type(torch.FloatTensor)
                p.data -= perturbation
    
            model_minus.train()
            output = model_plus(data)
            loss = F.nll_loss(output, target)
            loss.backward()
    
            g_plus = [p.grad.data for p in model_plus.parameters()]
            g_minus = [p.grad.data for p in model_minus.parameters()]
            Q = (self.model.flatten_array(g_plus) -
                 self.model.flatten_array(g_minus)) / (2 * self.Hess_est_r)
    
            val_grad = self.get_val_grad()
            self.model.UORO_update_step(Q)
            new_eta = self.model.get_updated_eta(self.mlr, val_grad=val_grad)
    
            # set_trace()
    
            for lr, eta in zip(self.optimizer.param_groups[0]['lr'], new_eta):
                lr.data = torch.from_numpy(eta).type(torch.FloatTensor)
            
        return training_loss

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


    def get_val_grad(self):
        val_model = copy(self.model)
        val_model.train()
        data, target = next(self.val_loader)
            
        output = val_model(data)
        test_loss = F.nll_loss(output, target)
        test_loss.backward()

        return [p.grad.data.numpy() for p in val_model.parameters()]
    
    def update_monitors(self):
        """Loops through the monitor keys and appends current value of any
        object's attribute found."""

        for key in self.mons:
            try:
                self.mons[key].append(rgetattr(self, key))
            except AttributeError:
                pass