import time

import torch
import numpy as np
from FEM import FEM


class Loss(torch.autograd.Function):
    # Late init parameters
    p = alpha = None
    VolumeFraction = cnt = N = 0
    x = network_input = Je = None
    mesh = fem = shape = frozen_mask = frozen = None
    init_objective = J = volume = 0

    @staticmethod
    def initialize(mesh, bc, frozen, opts, device):
        Loss.frozen, Loss.frozen_mask = frozen
        Loss.N = mesh['elem_num']
        Loss.mesh, Loss.shape = mesh, mesh['shape']
        Loss.fem = FEM(mesh, bc)
        Loss.p, Loss.alpha = opts['penalty'], opts['alpha']
        Loss.VolumeFraction = opts['volume_fraction']

        _, j = Loss.fem.solve(np.ones(Loss.N))
        Loss.init_objective = j.sum()
        network_input = j.clip(np.nanpercentile(j, 2), np.nanpercentile(j, 98))
        Loss.network_input = torch.tensor(network_input).float()[None, None, :, :].to(device)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        s = time.time()
        Loss.x = x[0][0].cpu().detach().numpy()
        print(time.time() - s)
        Loss.x[Loss.frozen_mask] = Loss.frozen[Loss.frozen_mask]
        Loss.volume = Loss.x.mean()

        # Objective
        e = (Loss.x ** Loss.p)
        _, Loss.Je = Loss.fem.solve(e.flatten(order='F'))
        Loss.J = (e * Loss.Je).sum()

        # Constraint
        Loss.cnt = Loss.volume - Loss.VolumeFraction

        return torch.tensor(Loss.J / Loss.init_objective + Loss.alpha * Loss.cnt ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors

        # Objective
        d_objective = - Loss.p * (Loss.x ** (Loss.p - 1)) * Loss.Je / Loss.init_objective

        # Constraint
        d_cnt = 2 * Loss.alpha * Loss.cnt * np.ones_like(Loss.x) / Loss.N

        d_loss = d_objective + d_cnt
        d_loss[Loss.frozen_mask] = 0
        return grad_output * torch.from_numpy(d_loss)[None, None, :, :].to(x.device)
