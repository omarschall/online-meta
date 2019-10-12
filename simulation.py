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
from copy import copy, deepcopy

class Simulation:

    def __init__(self, model, optimizer, monitors=[], Hess_est_r=0.001,
                 mlr=0.000001, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.monitors = monitors
        self.Hess_est_r = Hess_est_r
        self.mlr = mlr
        self.update_optimizer_online = True

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
            if self.model.name == 'Private_LR':
                A = self.model.unflatten_array(self.model.A)
            elif self.model.name == 'Global_LR':
                A = self.model.unflatten_array(self.model.Gamma)
        
            
            model_plus = deepcopy(self.model)
#            for p in model_plus.parameters():
#                print(p.data[0][0])
#                break
            for param, direction in zip(model_plus.parameters(), A):
                perturbation = torch.from_numpy(self.Hess_est_r * direction).type(torch.FloatTensor)
                param.data.add_(perturbation)
#            for p in model_plus.parameters():
#                print(p.data[0][0])
#                break
            model_plus.train()
            output = model_plus(data)
            loss = F.nll_loss(output, target)
            loss.backward()
    
            model_minus = deepcopy(self.model)
#            for p in model_minus.parameters():
#                print(p.data[0][0])
#                break
            for param, direction in zip(model_minus.parameters(), A):
                perturbation = torch.from_numpy(self.Hess_est_r * direction).type(torch.FloatTensor)
                param.data.add_(-perturbation)
#            for p in model_minus.parameters():
#                print(p.data[0][0])
#                break
            
            model_minus.train()
            output = model_minus(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            
#            for p in self.model.parameters():
#                print(p.data[0][0])
#                break
    
            #set_trace()
            
            g_plus = [p.grad.data for p in model_plus.parameters()]
            g_minus = [p.grad.data for p in model_minus.parameters()]
            Q = (self.model.flatten_array(g_plus) -
                 self.model.flatten_array(g_minus)) / (2 * self.Hess_est_r)
    
            #set_trace()
    
            val_grad = self.get_val_grad()
            self.model.update_Gamma(Q)
            self.model.update_eta(self.mlr, val_grad=val_grad)
    
            #Update optimizer with new eta
            if self.update_optimizer_online:
                self.update_optimizer_by_eta()
            
        return training_loss

    def test(self):
        self.model.eval()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                self.test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.test_loss /= len(self.test_loader.dataset)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        self.correct = correct

    def get_val_grad(self):
        val_model = deepcopy(self.model)
        val_model.train()
        data, target = next(self.val_loader)
            
        output = val_model(data)
        test_loss = F.nll_loss(output, target)
        test_loss.backward()

        return [p.grad.data.numpy() for p in val_model.parameters()]
    
    def update_optimizer_by_eta(self):
        """Updates the optimizer used by the simulation with the new eta values
        from the model."""
        if self.model.name == 'Private_LR':
            new_eta = self.model.unflatten_array(self.model.eta)
            for lr, eta in zip(self.optimizer.param_groups[0]['lr'], new_eta):
                lr.data = torch.from_numpy(eta).type(torch.FloatTensor)
            # Get means by layer
            self.layer_wise_means = []
            self.layer_wise_stds = []
            for i_param in range(0, len(new_eta), 2):
                mean = np.concatenate([new_eta[i_param].flatten(),
                                       new_eta[i_param+1].flatten()]).mean()
                self.layer_wise_means.append(mean)
                std = np.concatenate([new_eta[i_param].flatten(),
                                       new_eta[i_param+1].flatten()]).std()
                self.layer_wise_stds.append(std)
                
        elif self.model.name == 'Global_LR':
            self.optimizer.param_groups[0]['lr'] = np.copy(self.model.eta)
    
    def update_monitors(self):
        """Loops through the monitor keys and appends current value of any
        object's attribute found."""

        for key in self.mons:
            try:
                self.mons[key].append(rgetattr(self, key))
            except AttributeError:
                pass