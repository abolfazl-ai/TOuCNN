from numpy import nan


def get_example_bc(example_num):
    # Tip-loaded cantilever
    if example_num == 1:
        return [
            {'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': 0},
            {'S': (1, 0), 'E': (1, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}
        ], [], 0.75, None
    # Mid-loaded cantilever
    elif example_num == 2:
        return [
            {'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': 0},
            {'S': (1, 0.5), 'E': (1, 0.5), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}
        ], [], 0.50, None
    # MBB beam
    elif example_num == 3:
        return [
            {'S': (0, 0), 'E': (0, 1), 'D': (0, nan), 'F': 0},
            {'S': (1, 0), 'E': (1, 0), 'D': (nan, 0), 'F': 0},
            {'S': (0, 1), 'E': (0, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -1)},
        ], [], 0.45, None
    # Michell beam
    elif example_num == 4:
        return [
            {'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
            {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
            {'S': (0.5, 0), 'E': (0.5, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)},
        ], [], 0.30, 'Y'
    # Distributed load bridge
    elif example_num == 5:
        return [
            {'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
            {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
            {'S': (0, 1), 'E': (1, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -0.009)},
        ], [], 0.20, 'Y'
    # Bridge structure (Ordered SIMP example)
    elif example_num == 6:
        return [
            {'S': (0, 0), 'E': (0, 0), 'D': (nan, 0), 'F': 0},
            {'S': (1, 0), 'E': (1, 0), 'D': (nan, 0), 'F': 0},
            {'S': (0.50, 0), 'E': (0.50, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -2)},
            {'S': (0.25, 0), 'E': (0.25, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)},
            {'S': (0.75, 0), 'E': (0.75, 0), 'D': (nan, nan), 'F': lambda x, y: (0, -1)}
        ], [], 0.40, 'Y'
    # Compression
    elif example_num == 7:
        return [
            {'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
            {'S': (0, 1), 'E': (1, 1), 'D': (nan, nan), 'F': lambda x, y: (0, -0.1)}
        ], [
            {'S': (0, 0), 'E': (1, 0.02), 'Material': 'C'},
            {'S': (0, 0.98), 'E': (1, 1), 'Material': 'C'},
        ], 0.30, 'Y'
