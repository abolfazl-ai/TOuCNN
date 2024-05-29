import time

import numpy as np
from scipy.ndimage import uniform_filter
from FEM import FEM
from input_reader import get_inputs


class OCOptimizer:

    def __init__(self, bc, preserved, opts):
        self.x, self.opts = np.zeros(np.flip(opts['mesh_size'])), opts
        mesh, bc, frozen, self.sym = get_inputs(opts, bc, preserved)
        self.frozen, self.frozen_mask = frozen[0], ~frozen[1]
        self.shape, self.N, self.opts = mesh['shape'], mesh['elem_num'], opts
        self.p, self.move, self.r = opts['penalty'], opts['move'], self.opts['filter_radius']
        self.VolumeFraction = opts['volume_fraction']

        self.fem = FEM(mesh, bc)

        self.x = self.frozen.copy()
        self.x[self.frozen_mask] = self.VolumeFraction

        self.history = {'Objective': [], 'Volume': [], 'Convergence': [], 'penalty': [], 'Time': 0.0}

    def get_objective(self, x):
        e = x ** self.p
        u, Je = self.fem.solve(e.flatten(order='F'))
        J = (e * Je).sum()
        dJ = - self.p * (x ** (self.p - 1)) * Je
        return J, dJ

    def optimality_criteria(self, x, dc):
        x_new, xT = x.copy(), x[self.frozen_mask]
        xU, xL = xT + self.move, xT - self.move
        ocP = xT * np.real(np.sqrt(-dc[self.frozen_mask]))
        LM = [0, np.mean(ocP) / self.VolumeFraction]
        while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
            l_mid = 0.5 * (LM[0] + LM[1])
            x_new[self.frozen_mask] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
            LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > self.VolumeFraction else (LM[0], l_mid)
        return x_new

    def optimize(self, iter_callback):
        start_time = time.time()
        for it in range(self.opts['max_it']):
            x = uniform_filter(self.x, self.r, mode='reflect')
            x = x.clip(self.x - self.move, self.x + self.move).clip(1E-6, 1)
            J, dJ = self.get_objective(x)
            x = self.optimality_criteria(x, dJ)

            change = np.linalg.norm(x - self.x) / np.sqrt(self.N)
            grey = np.logical_and(0.1 < x, x < 0.9).mean()

            self.x[self.frozen_mask] = x[self.frozen_mask]

            self.history['Objective'].append(J)
            self.history['Volume'].append(self.x.mean())
            self.history['Convergence'].append(change)
            self.history['penalty'].append(self.p)

            iter_callback(it, self.x, self.history)

            self.p = min(self.p + self.opts['penalty_increase'], 8.0)
            if change < self.opts['converge_criteria'] and it > self.opts['min_it']: break

        self.history['Time'] = time.time() - start_time
        return self.x
