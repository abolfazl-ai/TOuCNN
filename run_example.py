from utils import optimize
from CNNOptimizer import CNNOptimizer
from examples import get_example_bc

bc, frozen, MF, sym = get_example_bc(2)

materials = {'name': ['Void', 'A', 'B', 'C'],  # Display name of material
             'color': ['w', 'b', 'r', 'k'],  # Display color of material
             'D': [0, 0.4, 0.7, 1],  # Normalized material density
             'E': [0, 0.2, 0.6, 1],  # Normalized material elastic modulus
             'C': [0, 0.5, 0.8, 1]}  # Normalized material cost

# Uncomment for single material
materials = {'name': ['Void', 'C'], 'color': ['w', 'k'], 'D': [0, 1], 'E': [0, 1], 'C': [0, 1]}

cnn_opts = {
    'use_gpu': True,
    'mesh_size': (150, 75),  # (nx, ny)
    'symmetry_axis': sym,  # Possible modes: None, 'X', 'Y'
    'main_constraint': 'M',  # V: Volume fraction, M: Mass fraction
    'main_factor': MF,
    'cost_factor': 0.50,
    'penalty': 1.0, 'gamma': 1.0075,  # Continuation scheme
    'alpha': 1.0, 'alpha_increase': 0.50,  # Main constraint penalty
    'beta': 1.0, 'beta_increase': 0.25,  # Cost constraint penalty
    'learning_rate': 1E-4,
    'min_it': 100, 'max_it': 500, 'converge_criteria': 0.005  # Convergence options
}

opt = CNNOptimizer(bc, materials, frozen, cnn_opts)
optimize(opt, materials)
