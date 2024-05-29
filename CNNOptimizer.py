import time

import torch
import random
import numpy as np
from torch import optim
from CNN import TopOptCNN
from Loss import Loss
from input_reader import get_inputs


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(0)


class CNNOptimizer:

    def __init__(self, bc, preserved, opts):
        self.x, self.opts = np.zeros(np.flip(opts['mesh_size'])), opts
        mesh, bc, frozen, self.sym = get_inputs(opts, bc, preserved)
        device = 'cuda' if (torch.cuda.is_available() and opts['use_gpu']) else 'cpu'
        Loss.initialize(mesh, bc, frozen, opts, device)

        self.history = {'Loss': [], 'Objective': [], 'Volume': [], 'Convergence': [],
                        'penalty': [], 'alpha': [], 'LearningRate': [], 'Time': 0.0}

        self.network = TopOptCNN().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), amsgrad=True, lr=opts['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, opts['step'], gamma=opts['gamma'])

    def optimize(self, iter_callback):
        start_time = time.time()
        for epoch in range(self.opts['max_it']):
            self.optimizer.zero_grad()
            x = self.network(Loss.network_input)

            loss = Loss.apply(x)
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.05)
            self.optimizer.step()
            self.scheduler.step()

            Loss.p = min(Loss.p + self.opts['penalty_increase'], 8.0)
            Loss.alpha = min(Loss.alpha + self.opts['alpha_increase'], 100.0)
            grey = np.logical_and(0.1 < Loss.x, Loss.x < 0.9).mean()

            self.history['Loss'].append(loss.item())
            self.history['Objective'].append(self.sym['factor'] * Loss.J)
            self.history['Volume'].append(Loss.volume)
            self.history['Convergence'].append(grey)
            self.history['penalty'].append(Loss.p)
            self.history['alpha'].append(Loss.alpha)
            self.history['LearningRate'].append(self.scheduler.get_last_lr()[0])

            self.x = Loss.x
            if self.opts['symmetry_axis'] is not None:
                self.x = np.concatenate((self.x, np.flip(self.x, self.sym['axis'])), self.sym['axis'])
            iter_callback(epoch, self.x, self.history)

            if grey < self.opts['converge_criteria'] and epoch > self.opts['min_it']: break

        self.history['Time'] = time.time() - start_time
        return self.x
