import time

import numpy as np
import torch
from torch import optim
from CNN import TopOptCNN
from Loss import Loss
from input_reader import get_inputs


def set_seed(seed):
    print(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(1234)


class CNNOptimizer:

    def __init__(self, bc, frozen, opts):
        self.x, self.opts = np.zeros(np.flip(opts['mesh_size'])), opts
        mesh, bc, frozen, self.sym = get_inputs(opts['symmetry_axis'], opts['mesh_size'], bc, frozen)
        device = 'cuda' if (torch.cuda.is_available() and opts['use_gpu']) else 'cpu'
        Loss.initialize(mesh, bc, frozen, opts, device)

        self.history = {'Loss': [], 'Objective': [], 'Volume': [], 'Gray': [],
                        'Penalty': [], 'Alpha': []}

        self.network = TopOptCNN().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), amsgrad=True, lr=opts['learning_rate'])

    def optimize(self, iter_callback):
        start_time = time.time()
        for epoch in range(self.opts['max_it']):
            self.optimizer.zero_grad()
            x = self.network(Loss.network_input)
            loss = Loss.apply(x)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.05)
            self.optimizer.step()

            Loss.p = min(Loss.p + self.opts['penalty_increase'], 8.0)
            Loss.alpha = Loss.alpha + self.opts['alpha_increase']
            grey = np.logical_and(0.1 < Loss.x, Loss.x < 0.9).mean()

            self.history['Loss'].append(loss.item())
            self.history['Objective'].append(self.sym['factor'] * Loss.J)
            self.history['Volume'].append(Loss.volume)
            self.history['Gray'].append(grey)
            self.history['Penalty'].append(Loss.p)
            self.history['Alpha'].append(Loss.alpha)

            self.x = Loss.x
            if self.opts['symmetry_axis'] is not None:
                self.x = np.concatenate((Loss.x, np.flip(Loss.x, self.sym['axis'])), self.sym['axis'])
            iter_callback(self.x, self.history)

            if grey < self.opts['converge_criteria'] and epoch > self.opts['min_it']: break

        return self.x, time.time() - start_time
