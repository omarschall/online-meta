"""main.py

Author: @omarschall, 9-13-2019"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from pdb import set_trace
from copy import copy
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from utils import *
from network import Net, Global_LR_Net
from optimizers import SGD_Multi_LR
from simulation import Simulation
import os
from itertools import product
import pickle

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 10
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    #LRs = list(np.linspace(0.001, 0.45, 20))
    LRs = list(np.logspace(-4, 0, 20))[13:18]
    #MLRs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    macro_configs = config_generator(lr=LRs)
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job // n_seeds
    torch.manual_seed(i_job)
    np.random.seed(i_job+1)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    MNIST_path = '/scratch/oem214/online-meta/data/'

if os.environ['HOME'] == '/Users/omarschall':
    params = {'lr': 0.01}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #torch.manual_seed(3)
    #np.random.seed(4)
    
    MNIST_path = './data/'

params['mode'] = 'private'
params['mlr'] = 0.001
#params['lr'] = 0.03
params['r'] = 0.00001 
params['n_train_epochs'] = 3
params['n_test_epochs'] = 3

meta_test_size = 10000
batch_size = 200
val_batch_size = 200
test_batch_size = 5000
mlr = params['mlr']
n_epochs = params['n_train_epochs']
test_epochs = params['n_test_epochs']
lr = params['lr']
r = params['r']

### --- Set up training data --- ###

all_train_data = datasets.MNIST(MNIST_path, train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
meta_train_set, meta_test_set = torch.utils.data.random_split(all_train_data, [60000 - meta_test_size,
                                                                               meta_test_size])
meta_train_loader = torch.utils.data.DataLoader(meta_train_set, batch_size=batch_size, shuffle=False)
meta_test_loader = torch.utils.data.DataLoader(meta_test_set, batch_size=batch_size, shuffle=False)

### --- Set up test data --- ###

all_test_data = datasets.MNIST(MNIST_path, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
val_set, test_set = torch.utils.data.random_split(all_test_data, [10000 - test_batch_size,
                                                                  test_batch_size])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

layer_sizes = [784, 1000, 500, 200, 10]
params['layer_sizes'] = layer_sizes

init_params = init_random_MLP_params(layer_sizes)

if params['mode'] == 'private':
    model = Net(layer_sizes, lr_init=lr)
if params['mode'] in ['global', 'baseline']:
    model = Global_LR_Net(layer_sizes, lr)
for p, p_init in zip(model.parameters(), init_params):
    p.data = torch.from_numpy(p_init).type(torch.FloatTensor)
if params['mode'] == 'private':
    optimizer = SGD_Multi_LR(model.parameters(), lr_init=lr)
if params['mode'] in ['global', 'baseline']:
    optimizer = optim.SGD(model.parameters(), lr=lr)

test_losses = {}

if params['mode'] in ['global', 'private']:
    print('\nMeta training test loss \n'.upper())
    
    #Train sim
    monitors = ['model.H_norm', 'model.grad_norm', 'model.Gamma_norm']
    if params['mode'] == 'global':
        monitors.append('model.eta')
    elif params['mode'] == 'private':
        monitors = monitors + ['model.mean_eta', 'layer_wise_means', 'layer_wise_stds']
    
    sim = Simulation(model, optimizer, monitors=monitors, Hess_est_r=r,  mlr=mlr,
                     update_optimizer_online=True)
    sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader, n_epochs=n_epochs)
    
    sim.update_optimizer_by_eta()
    
    test_losses['meta_training_test_loss'] = sim.test_loss
    test_losses['meta_training_test_correct'] = sim.correct

if params['mode'] in ['global', 'private']:
    print('\nMeta test test loss \n'.upper())
    
    #Test sim with fixed optimizer
    #Re-init parameters
    for p, p_init in zip(model.parameters(), init_params):
        p.data = torch.from_numpy(p_init).type(torch.FloatTensor)
    test_sim = Simulation(model, sim.optimizer)
    test_sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader,
                 mode='test', n_epochs=test_epochs)
    
    test_losses['meta_test_test_loss'] = test_sim.test_loss
    test_losses['meta_test_test_correct'] = test_sim.correct

if params['mode'] == 'baseline':
    print('\nBaseline init LR loss \n'.upper())
    
    #Test sim with basic optimzer
    fixed_LR_optimizer = optim.SGD(model.parameters(), lr=lr)
    for p, p_init in zip(model.parameters(), init_params):
        p.data = torch.from_numpy(p_init).type(torch.FloatTensor)
    test_sim = Simulation(model, fixed_LR_optimizer)
    test_sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader,
                 mode='test', n_epochs=test_epochs)
    
    test_losses['init_LR_test_loss'] = test_sim.test_loss
    test_losses['init_LR_test_correct'] = test_sim.correct

if False:
    print('\nBaseline mean LR loss \n'.upper())
    
    #Test sim with basic optimzer
    mean_lr = model.eta.mean()
    fixed_LR_optimizer = SGD_Multi_LR(model.parameters(), lr_init=mean_lr)
    for p, p_init in zip(model.parameters(), init_params):
        p.data = torch.from_numpy(p_init).type(torch.FloatTensor)
    test_sim = Simulation(model, fixed_LR_optimizer)
    test_sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader,
                 mode='test', n_epochs=test_epochs)
    
    test_losses['mean_LR_test_loss'] = test_sim.test_loss
    test_losses['mean_LR_test_correct'] = test_sim.correct

if False:
    print('\nBaseline test with ADAM \n'.upper())
    
    #Test sim with ADAM optimzer
    ADAM_optimizer = optim.Adam(model.parameters(), lr=lr)
    for p, p_init in zip(model.parameters(), init_params):
        p.data = torch.from_numpy(p_init).type(torch.FloatTensor)
    test_sim = Simulation(model, ADAM_optimizer)
    test_sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader,
                 mode='test', n_epochs=test_epochs)
    
    test_losses['Adam_test_loss'] = test_sim.test_loss
    test_losses['Adam_test_correct'] = test_sim.correct

if os.environ['HOME'] == '/Users/omarschall':
    #pass
    mean_LRs = [float(LR.mean()) for LR in sim.optimizer.param_groups[0]['lr']]
    std_LRs = [float(LR.std()) for LR in sim.optimizer.param_groups[0]['lr']]
    LR = sim.optimizer.param_groups[0]['lr']
    mean_layer_concats = [np.concatenate([LR[i].flatten(),
                                          LR[i+1].flatten()]).mean() for i in [0, 2, 4, 6]]
    std_layer_concats = [np.concatenate([LR[i].flatten(),
                                         LR[i+1].flatten()]).std()/np.sqrt(LR[i].numel() + LR[i+1].numel()) for i in [0, 2, 4, 6]]
    fig = plt.figure(figsize=(2.5, 2.5))
    plt.errorbar(range(len(mean_layer_concats)), mean_layer_concats, yerr=std_layer_concats)
    #plt.errorbar(range(len(mean_LRs)), mean_LRs, yerr=std_LRs)
    plt.axhline(y=lr, color='C1')
    #param_types = ['W', 'b']
    #labels = ['Layer-{} {}'.format(i // 2, param_types[i % 2]) for i in range(len(mean_LRs))]
    #plt.xticks(range(len(mean_LRs)))#, labels)
    #plt.legend(['Init LR', 'Final LR'])
    # plt.ylim([0,0.1])
#    
#    plt.figure()
#    for i_LR, LR in enumerate(sim.optimizer.param_groups[0]['lr']):
#        LR_ = LR.data.numpy().flatten()
#        plt.plot(np.ones_like(LR_)*i_LR, LR_, '.', alpha=2/np.sqrt(len(LR_)))
#    plt.xticks(range(i_LR+1), labels)


if os.environ['HOME'] == '/home/oem214':
    
    ### --- PACKAGE DATA --- ###
    
    #mean_LRs = [float(LR.mean()) for LR in sim.optimizer.param_groups[0]['lr']]
    #std_LRs = [float(LR.std()) for LR in sim.optimizer.param_groups[0]['lr']]
    #param_types = ['W', 'b']
    #labels = ['Layer-{} {}'.format(i // 2, param_types[i % 2]) for i in range(len(mean_LRs))]

    #LRs = [LR.data.numpy() for LR in sim.optimizer.param_groups[0]['lr']]

#    mean_LRs = None
#    std_LRs = None
#    labels = None
#    mean_lr = None
#
#    {'mean_LRs': mean_LRs, 'std_LRs': std_LRs, 'labels': labels,
#     'global_mean_lr': mean_lr}

    result = {'config': params, 'i_config': i_config, 'i_job': i_job}
    if params['mode'] in ['global', 'private']:
        result['monitors'] = sim.mons
    result.update(test_losses)
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_' + str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
