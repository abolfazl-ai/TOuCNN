from numpy import nan
from utils import optimize
from CNNOptimizer import CNNOptimizer

# Boundary conditions and loads
# S: Starting point, E: End point, D: Displacement, F: Force
bc = [
    {'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
    {'S': (0, 1), 'E': (1, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -0.1)}
]

# Non-design regions
# S: Starting point, E: End point, Material: Fixed material
frozen = [
    {'S': (0, 0), 'E': (1, 0.02), 'Material': 'C'},
    {'S': (0, 0.98), 'E': (1, 1), 'Material': 'C'},
]

materials = {'name': ['Void', 'A', 'B', 'C'],  # Display name of material
             'color': ['w', 'b', 'r', 'k'],  # Display color of material
             'D': [0, 0.4, 0.7, 1],  # Normalized material density
             'E': [0, 0.2, 0.6, 1],  # Normalized material elastic modulus
             'C': [0, 0.5, 0.8, 1]}  # Normalized material cost

# Uncomment for single material
# materials = {'name': ['Void', 'C'], 'color': ['w', 'k'], 'D': [0, 1], 'E': [0, 1], 'C': [0, 1]}

cnn_opts = {
    'use_gpu': True,
    'mesh_size': (256, 256),  # (nx, ny)
    'symmetry_axis': 'Y',  # Possible modes: None, 'X', 'Y'
    'main_constraint': 'M',  # V: Volume fraction, M: Mass fraction
    'main_factor': 0.3,
    'cost_factor': 1,
    'penalty': 1, 'penalty_increase': 0.05,  # Continuation scheme
    'alpha': 1, 'alpha_increase': 2.0,  # Main constraint penalty
    'beta': 1, 'beta_increase': 0.5,  # Cost constraint penalty
    'learning_rate': 1E-4,
    'min_it': 100, 'max_it': 500, 'converge_criteria': 0.005  # Convergence options
}

opt = CNNOptimizer(bc, materials, frozen, cnn_opts)
optimize(opt, materials)
