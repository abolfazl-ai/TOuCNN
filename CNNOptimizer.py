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

    def __init__(self, bc, materials, preserved, opts):
        self.x, self.opts = np.zeros(np.flip(opts['mesh_size'])), opts
        mesh, bc, frozen, self.sym = get_inputs(opts, bc, preserved, materials)
        Loss.initialize(materials, mesh, bc, frozen, opts)

        self.history = {'Loss': [], 'Objective': [], 'Volume': [], 'Mass': [], 'Cost': [],
                        'GreyElements': [], 'penalty': [], 'alpha': [], 'beta': []}

        self.device = 'cuda' if (torch.cuda.is_available() and opts['use_gpu']) else 'cpu'
        self.network = TopOptCNN(Loss.M).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), amsgrad=True, lr=opts['learning_rate'])

    def optimize(self, iter_callback):
        for epoch in range(self.opts['max_it']):
            self.optimizer.zero_grad()
            x = self.network(Loss.network_input.to(self.device))
            loss = Loss.apply(x)
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.01)
            self.optimizer.step()

            Loss.p = min(Loss.p * self.opts['gamma'], 8.0)
            Loss.alpha = min(Loss.alpha + self.opts['alpha_increase'], 100.0)
            Loss.beta = min(Loss.beta + self.opts['beta_increase'], 100.0)
            grey = torch.logical_and(0.2 < x, x < 0.8).float().mean().cpu().item()

            self.history['Loss'].append(loss.item())
            self.history['Objective'].append(self.sym['factor'] * Loss.objective)
            self.history['Volume'].append(Loss.volume)
            self.history['Mass'].append(Loss.mass)
            self.history['Cost'].append(Loss.H)
            self.history['GreyElements'].append(grey)
            self.history['penalty'].append(Loss.p)
            self.history['alpha'].append(Loss.alpha)
            self.history['beta'].append(Loss.beta)

            self.x = Loss.density
            if self.opts['symmetry_axis'] is not None:
                self.x = np.concatenate((self.x, np.flip(self.x, self.sym['axis'])), self.sym['axis'])
            iter_callback(epoch, self.x, self.history)

            if grey < self.opts['converge_criteria'] and epoch > self.opts['min_it']: break
        return self.x
