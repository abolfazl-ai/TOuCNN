from numpy import nan


def get_example(example_num, use_gpu=True):
    name, bc, frozen = '', [], []
    cnn_opts = {
        'use_gpu': use_gpu,
        'mesh_size': (256, 128),
        'symmetry_axis': None,
        'alpha_increase': 5,
        'penalty': 3, 'penalty_increase': 0.02,
        'min_it': 25, 'max_it': 500, 'converge_criteria': 0.02
    }

    if example_num == 1:
        name = 'Tip-loaded cantilever'
        bc = [{'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': 0},
              {'S': (1, 0), 'E': (1, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}]
        cnn_opts['volume_fraction'] = 0.75
        cnn_opts['learning_rate'] = 2.5E-4

    elif example_num == 2:
        name = 'Mid-loaded cantilever'
        bc = [{'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': 0},
              {'S': (1, 0.5), 'E': (1, 0.5), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}]
        cnn_opts['volume_fraction'] = 0.50
        cnn_opts['learning_rate'] = 1.5E-4

    elif example_num == 3:
        name = 'MBB beam'
        bc = [{'S': (0, 0), 'E': (0, 1), 'D': (0, nan), 'F': 0},
              {'S': (1, 0), 'E': (1, 0), 'D': (nan, 0), 'F': 0},
              {'S': (0, 1), 'E': (0, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}]
        cnn_opts['volume_fraction'] = 0.45
        cnn_opts['learning_rate'] = 2.0E-4

    elif example_num == 4:
        name = 'Michell beam'
        bc = [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
              {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
              {'S': (0.5, 0), 'E': (0.5, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}]
        cnn_opts['volume_fraction'] = 0.30
        cnn_opts['learning_rate'] = 2.0E-4
        cnn_opts['symmetry_axis'] = 'Y'

    elif example_num == 5:
        name = 'Distributed load bridge'
        bc = [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
              {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
              {'S': (0, 1), 'E': (1, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -0.01)}]
        cnn_opts['volume_fraction'] = 0.20
        cnn_opts['learning_rate'] = 1.0E-4
        cnn_opts['symmetry_axis'] = 'Y'

    elif example_num == 6:
        name = 'Bridge structure'
        bc = [{'S': (0, 0), 'E': (0, 0), 'D': (nan, 0), 'F': 0},
              {'S': (1, 0), 'E': (1, 0), 'D': (nan, 0), 'F': 0},
              {'S': (0.50, 0), 'E': (0.50, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -2)},
              {'S': (0.25, 0), 'E': (0.25, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)},
              {'S': (0.75, 0), 'E': (0.75, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}]
        cnn_opts['volume_fraction'] = 0.40
        cnn_opts['learning_rate'] = 2.5E-4
        cnn_opts['symmetry_axis'] = 'Y'

    return name, bc, frozen, cnn_opts
