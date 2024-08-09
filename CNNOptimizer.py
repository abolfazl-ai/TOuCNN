import time
import numpy as np
import torch
import random
from torch import optim
from CNN import TopOptCNN
from Loss import Loss
from input_reader import get_inputs


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


class CNNOptimizer:

    def __init__(self, bc, frozen, opts):
        set_seed(1234)
        self.history = {'Loss': [], 'Objective': [], 'Volume': [],
                        'Gray': [], 'Penalty': [], 'Alpha': []}

        # Initialize the design variable and options
        self.x, self.opts = np.zeros(np.flip(opts['mesh_size'])), opts

        # Get the mesh, boundary conditions, frozen nodes, and symmetry settings
        mesh, bc, frozen, self.sym = get_inputs(opts, bc, frozen)

        # Initialize the loss function with provided parameters
        device = 'cuda' if (torch.cuda.is_available() and opts['use_gpu']) else 'cpu'
        Loss.initialize(mesh, bc, frozen, opts, device)

        # Initialize the CNN and ADAM optimizer
        self.network = TopOptCNN().to(device)
        self.optimizer = optim.Adam(self.network.parameters(), amsgrad=True, lr=opts['learning_rate'])

    def optimize(self, iter_callback):
        start_time = time.time()
        for epoch in range(self.opts['max_it']):
            self.optimizer.zero_grad()

            # Forward pass through the network to get the design variable
            x = self.network(Loss.network_input)

            # Compute the loss
            loss = Loss.apply(x)

            # Backward pass to compute gradients
            loss.backward(retain_graph=True)

            # Clip the gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)

            # Update the network parameters
            self.optimizer.step()

            # Update penalty and alpha values
            Loss.p = min(Loss.p + self.opts['penalty_increase'], 8.0)
            Loss.alpha = Loss.alpha + self.opts['alpha_increase']

            # Calculate the proportion of intermediate values in the design
            grey = np.logical_and(0.1 < Loss.x, Loss.x < 0.9).mean()

            # Record the loss and other metrics
            self.history['Loss'].append(loss.item())
            self.history['Objective'].append(self.sym['multiplier'] * Loss.J)
            self.history['Volume'].append(Loss.volume)
            self.history['Gray'].append(grey)
            self.history['Penalty'].append(Loss.p)
            self.history['Alpha'].append(Loss.alpha)

            # Update the design variable
            self.x = Loss.x

            # Apply symmetry if specified
            if self.opts['symmetry_axis'] is not None:
                self.x = np.concatenate((Loss.x, np.flip(Loss.x, self.sym['axis'])), self.sym['axis'])

            # Call the iteration callback function with the current design and history
            iter_callback(self.x, self.history)

            # Check for convergence based on the grey level and minimum iterations
            if grey < self.opts['converge_criteria'] and epoch > self.opts['min_it']: break

        # Return the final design and the total optimization time
        return self.x, time.time() - start_time
