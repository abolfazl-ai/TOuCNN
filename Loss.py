import torch
import numpy as np
from FEM import FEM


class Loss(torch.autograd.Function):
    # Late init parameters
    p = alpha = beta = None
    E = D = C = C_min = V = N = M = None
    network_input = je = density = None
    main_cnt = MF = use_volume = cost_cnt = cost = H = CF = 0
    mesh = fem = shape = frozen_mask = frozen = None
    init_objective = objective = volume = mass = 0

    @staticmethod
    def initialize(materials, mesh, bc, frozen, opts):
        Loss.frozen, Loss.frozen_mask = frozen
        Loss.E = np.expand_dims(materials['E'], axis=(0, 2, 3))
        Loss.D = np.expand_dims(materials['D'], axis=(0, 2, 3))
        Loss.C = np.expand_dims(materials['C'], axis=(0, 2, 3))
        Loss.N, Loss.M = mesh['elem_num'], len(materials['name'])
        Loss.mesh, Loss.shape = mesh, mesh['shape']
        Loss.fem = FEM(mesh, bc)
        Loss.p, Loss.alpha, Loss.beta = opts['penalty'], opts['alpha'], opts['beta']
        Loss.MF, Loss.CF = opts['main_factor'], opts['cost_factor']

        Loss.use_volume = opts['main_constraint'].upper() == 'V'
        Loss.V = np.ones_like(Loss.D)
        Loss.V[0][0] = 0
        Loss.C_min = sorted(materials['C'])[1]

        e = np.ones(Loss.shape) * (Loss.MF ** Loss.p)
        j = e * Loss.compliance(e)
        Loss.init_objective = j.sum()
        network_input = j.clip(np.nanpercentile(j, 2), np.nanpercentile(j, 98))
        Loss.network_input = torch.tensor(network_input).float()[None, None, :, :]

    @staticmethod
    def compliance(e):
        u = Loss.fem.solve(e.flatten(order='F'))
        je = ((u[Loss.mesh['c_mat']] @ Loss.fem.k) * u[Loss.mesh['c_mat']]).sum(1)
        return je.reshape(Loss.shape, order='F')

    @staticmethod
    def forward(ctx, x):
        # x = torchvision.transforms.GaussianBlur(15, sigma=0.1)(x)
        ctx.save_for_backward(x)
        x_np = x.cpu().detach().numpy()
        x_np[Loss.frozen_mask] = Loss.frozen[Loss.frozen_mask]
        Loss.density = (Loss.D * x_np)[0].sum(0)
        Loss.volume = (Loss.V * x_np).sum(1).mean()
        Loss.mass = Loss.density.mean()

        # Objective
        e = (Loss.E * (x_np ** Loss.p))[0].sum(0)
        Loss.je = Loss.compliance(e)
        Loss.objective = (e * Loss.je).sum()

        # Main constraint
        G = Loss.volume if Loss.use_volume else Loss.mass
        Loss.main_cnt = G - Loss.MF

        # Cost constraint
        if Loss.M > 2:
            Loss.cost = (Loss.C * x_np).sum(1).mean()
            Loss.H = (Loss.cost / Loss.volume - Loss.C_min) / (1 - Loss.C_min)
            Loss.cost_cnt = max(Loss.H - Loss.CF, 0)
        else:
            Loss.H = 1
            Loss.cost_cnt = 0

        return torch.tensor(Loss.objective / Loss.init_objective +
                            Loss.alpha * Loss.main_cnt ** 2 +
                            Loss.beta * Loss.cost_cnt ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x_np = x.cpu().detach().numpy()
        x_np[Loss.frozen_mask] = Loss.frozen[Loss.frozen_mask]

        # Objective
        d_objective = (Loss.E * (x_np ** (Loss.p - 1)))
        d_objective = - Loss.p * d_objective * Loss.je / Loss.init_objective

        # Main constraint
        d_main_cnt = 2 * Loss.alpha * Loss.main_cnt * (Loss.V if Loss.use_volume else Loss.D) / Loss.N

        # Cost constraint
        if Loss.M > 2:
            d_cost_cnt = (Loss.C * Loss.volume - Loss.cost * Loss.V) / (Loss.N * Loss.volume ** 2)
            d_cost_cnt = 2 * Loss.beta * Loss.cost_cnt * d_cost_cnt / (1 - Loss.C_min)
        else:
            d_cost_cnt = 0

        d_loss = grad_output * torch.tensor(d_objective + d_main_cnt + d_cost_cnt, device=x.device)
        d_loss[Loss.frozen_mask] = 0
        return d_loss
