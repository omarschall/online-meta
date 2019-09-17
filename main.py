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
import matplotlib.pyplot as plt
from utils import *
from network import Net
from optimizers import SGD_Multi_LR
from simulation import Simulation
import os

if os.environ['HOME'] == '/home/oem214':
    n_seeds = 20
    try:
        i_job = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    except KeyError:
        i_job = 0
    macro_configs = config_generator(mlr=[0.0001, 0.00001])
    micro_configs = tuple(product(macro_configs, list(range(n_seeds))))

    params, i_seed = micro_configs[i_job]
    i_config = i_job // n_seeds
    np.random.seed(i_job)

    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

if os.environ['HOME'] == '/Users/omarschall':
    params = {}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    np.random.seed(0)

meta_test_size = 10000
batch_size = 200
val_batch_size = 200
test_batch_size = 5000
n_epochs = 1
lr = 5e-2

### --- Set up training data --- ###

all_train_data = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
meta_train_set, meta_test_set = torch.utils.data.random_split(all_train_data, [60000 - meta_test_size,
                                                                               meta_test_size])
meta_train_loader = torch.utils.data.DataLoader(meta_train_set, batch_size=batch_size, shuffle=True)
meta_test_loader = torch.utils.data.DataLoader(meta_test_set, batch_size=batch_size, shuffle=True)

### --- Set up test data --- ###

all_test_data = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
val_set, test_set = torch.utils.data.random_split(all_test_data, [10000 - test_batch_size,
                                                                  test_batch_size])
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=val_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=test_batch_size, shuffle=True)


layer_sizes = [784, 1000, 200, 10]

init_params = init_random_MLP_params(layer_sizes)

model = Net([784, 1000, 200, 10], lr_init=lr)
for p, p_init in zip(model.parameters(), init_params):
    p.data = torch.from_numpy(p_init)
optimizer = SGD_Multi_LR(model.parameters(), lr_init=lr)

sim = Simulation(model, optimizer, monitors=[], Hess_est_r=0.001,  mlr=0.0001)
sim.run(meta_train_loader, meta_test_loader, val_loader, test_loader, n_epochs=n_epochs)

mean_LRs = [float(LR.mean()) for LR in sim.optimizer.param_groups[0]['lr']]
std_LRs = [float(LR.std()) for LR in sim.optimizer.param_groups[0]['lr']]
plt.errorbar(range(len(mean_LRs)), mean_LRs, yerr=std_LRs)
plt.axhline(y=lr, color='C1')
param_types = ['W', 'b']
labels = ['Layer-{} {}'.format(i // 2, param_types[i % 2]) for i in range(len(mean_LRs))]
plt.xticks(range(len(mean_LRs)), labels)
plt.legend(['Init LR', 'Final LR'])
# plt.ylim([0,0.1])

plt.figure()
for i_LR, LR in enumerate(sim.optimizer.param_groups[0]['lr']):
    LR_ = LR.data.numpy().flatten()
    plt.plot(np.ones_like(LR_)*i_LR, LR_, '.', alpha=2/np.sqrt(len(LR_)))
plt.xticks(range(i_LR+1), labels)


if os.environ['HOME'] == '/home/oem214':

    result = {'sim': sim, 'i_seed': i_seed, 'task': task,
              'config': params, 'i_config': i_config, 'i_job': i_job,
              'processed_data': processed_data}
    save_dir = os.environ['SAVEPATH']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'rnn_' + str(i_job))

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
