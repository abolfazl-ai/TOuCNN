import numpy as np
from numpy import nan
from cvxopt import matrix

"""
Utility functions for meshing the model, applying symmetry condition and
extracting boundary conditions and frozen regions from user inputs.
"""


def get_inputs(opts, bc, frozen):
    bc, frozen, shape, multiplier, axis = apply_symmetry(opts, bc, frozen)
    mesh = generate_mesh(*shape)
    bc = get_bc(mesh, bc)
    frozen = get_frozen(mesh, frozen)
    return mesh, bc, frozen, {'multiplier': multiplier, 'axis': axis}


def generate_mesh(nx, ny):
    shape = np.array((ny, nx))
    node_numbers = np.arange(np.prod(shape + 1)).reshape(shape + 1, order='F')
    elem_num, dof = np.prod(shape), np.prod(shape + 1) * 2  # Number of elements and degrees of freedom

    additions = np.array([0, 1, 2 * ny + 2, 2 * ny + 3, 2 * ny + 0, 2 * ny + 1, -2, -1])

    c_vec = np.reshape(2 * node_numbers[0:-1, 0:-1] + 2, (elem_num, 1), order='F')
    c_mat = c_vec + additions
    sI = np.hstack([np.arange(j, 2 * 4) for j in range(2 * 4)])
    sII = np.hstack([np.tile(j, 2 * 4 - j) for j in range(2 * 4)])
    iK, jK = c_mat[:, sI].T, c_mat[:, sII].T
    # Indexes of the global stiffness matrix
    indexes = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]

    return {'shape': shape, 'dof': dof, 'node_numbers': node_numbers, 'elem_num': elem_num,
            'indexes': (indexes[:, 0], indexes[:, 1]), 'c_mat': c_mat}


def get_frozen(mesh, preserved_regions):
    frozen, frozen_mask = np.zeros(mesh['shape']), np.zeros(mesh['shape'], dtype=bool)
    for row in preserved_regions:
        start, end, name = [row[key] for key in ('S', 'E', 'Material')]

        # Start and end node numbers
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1) for n, s, e
              in list(zip(mesh['shape'], *[(p[1], p[0]) for p in (start, end)]))]

        frozen_mask[nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]] = True
        frozen[nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]] = 1 if name.upper() == 'SOLID' else 0
    return frozen, frozen_mask


def get_bc(mesh, bc):
    fixed, dof = [], mesh['dof']
    force_vector = matrix(0.0, (dof, 1))
    for row in bc:
        start, end, displacement, force = [row[key] for key in ('S', 'E', 'D', 'F')]

        # Start and end node numbers
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e
              in list(zip(mesh['shape'], *[(p[1], p[0]) for p in (start, end)]))]
        nodes = (mesh['node_numbers'][nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]]).flatten()

        # Adding fix degrees of freedom
        fixed.extend(np.c_[2 * nodes, 2 * nodes + 1][:, np.argwhere(~np.isnan(displacement))].flatten().tolist())

        if callable(force):  # If the force vector is not zero
            for node in nodes:
                # Finding the x and y coordinates of nodes
                coordinates = np.flip(np.argwhere(mesh['node_numbers'] == node)[0] / mesh['shape'])
                force_vector[(2 * node + np.array([0, 1])).tolist(), 0] = force(*coordinates)

    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'fix_dofs': fixed, 'force_vector': force_vector}


def apply_symmetry(opts, boundary_conditions, frozen_regions):
    symmetry_axis, shape = opts['symmetry_axis'], list(opts['mesh_size'])
    if symmetry_axis is not None:
        axis = {'X': 1, 'Y': 0}[symmetry_axis.upper()]
        shape[axis] //= 2

        # Additional symmetry boundary conditions
        fixed = [{'S': (1, 0), 'E': (1, 1), 'D': (0, nan), 'F': 0},  # Symmetry axis: Y
                 {'S': (0, 1), 'E': (1, 1), 'D': (nan, 0), 'F': 0}]  # Symmetry axis: X
        bc = [fixed[axis]]

        # Modifying the boundary conditions
        for row in boundary_conditions:
            start, end, d, force = [row[key] for key in ('S', 'E', 'D', 'F')]
            s, e = list(start), list(end)
            s[axis], e[axis] = min(2 * s[axis], 1), min(2 * e[axis], 1)
            if callable(force):
                force = ForceTranslator(force, axis).new_force
            if start[axis] <= 0.5 or end[axis] <= 0.5:
                bc.append({'S': s, 'E': e, 'D': d, 'F': force})

        # Modifying the frozen regions
        frozen = []
        for row in frozen_regions:
            start, end, material = [row[key] for key in ('S', 'E', 'Material')]
            s, e = list(start), list(end)
            s[axis], e[axis] = min(2 * s[axis], 1), min(2 * e[axis], 1)
            if start[axis] <= 0.5 or end[axis] <= 0.5:
                frozen.append({'S': s, 'E': e, 'Material': material})

        return bc, frozen, shape, 2, 1 - axis

    return boundary_conditions, frozen_regions, shape, 1, None


class ForceTranslator:
    def __init__(self, force, axis):
        self.f, self.axis = force, axis

    def new_force(self, x, y):
        multiplier = 0.5 if (x, y)[self.axis] == 1 else 1
        [x, y][self.axis] *= 0.5
        f = np.array(self.f(x, y))
        return (f * multiplier).tolist()
