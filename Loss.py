import torch
import numpy as np
from FEM import FEM


class Loss(torch.autograd.Function):
    # Class-level attributes for storing various parameters and intermediate results
    p = alpha = 0
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
        Loss.p, Loss.alpha = opts['penalty'], opts['alpha_increase']
        Loss.VolumeFraction = opts['volume_fraction']

        # Initializing the network input using uniform material distribution
        _, je = Loss.fem.solve(np.ones(Loss.N))
        Loss.init_objective = je.sum()
        network_input = je.clip(np.nanpercentile(je, 2), np.nanpercentile(je, 98))  # Deleting outliers
        Loss.network_input = torch.tensor(network_input).float()[None, None, :, :].to(device)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        Loss.x = x[0][0].detach().cpu().numpy()

        # Applying non-design regions constraint
        Loss.x[Loss.frozen_mask] = Loss.frozen[Loss.frozen_mask]

        # Objective
        e = (Loss.x ** Loss.p)
        _, Loss.Je = Loss.fem.solve(e.flatten(order='F'))
        Loss.J = (e * Loss.Je).sum()

        # Constraint
        Loss.volume = Loss.x.mean()
        Loss.cnt = max(0, Loss.volume - Loss.VolumeFraction)

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
