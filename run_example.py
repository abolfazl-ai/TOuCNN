from utils import optimize
from CNNOptimizer import CNNOptimizer
from OCOptimizer import OCOptimizer
from examples import get_example_bc

bc, frozen, cons, sym = get_example_bc(1)

cnn_opts = {
    'use_gpu': True,
    'mesh_size': (128, 64),  # (nx, ny)
    'volume_fraction': cons,
    'symmetry_axis': sym,  # Possible modes: None, 'X', 'Y'
    'penalty': 3.0, 'penalty_increase': 0.05,  # Continuation scheme
    'alpha': 0.0, 'alpha_increase': 1.0,  # Main constraint penalty
    'learning_rate': 5E-4, 'gamma': 0.95, 'step': 5,
    'min_it': 10, 'max_it': 500, 'converge_criteria': 0.02  # Convergence options
}

opt = CNNOptimizer(bc, frozen, cnn_opts)
optimize(opt)

oc_opts = {
    'mesh_size': (128, 64),  # (nx, ny)
    'volume_fraction': cons,
    'symmetry_axis': sym,  # Possible modes: None, 'X', 'Y'
    'penalty': 3.0, 'penalty_increase': 0.025,  # Continuation scheme
    'move': 0.2, 'filter_radius': 3,
    'min_it': 10, 'max_it': 500, 'converge_criteria': 0.001  # Convergence options
}

opt = OCOptimizer(bc, frozen, oc_opts)
optimize(opt)
