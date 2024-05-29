import numpy as np
from numpy import nan
from cvxopt import matrix


def get_inputs(opts, bc, preserved):
    bc, shape, factor, axis = apply_symmetry(opts['symmetry_axis'], bc, opts['mesh_size'])
    mesh = generate_mesh(*shape)
    bc = get_bc(mesh, bc)
    frozen = get_frozen(mesh, preserved)
    return mesh, bc, frozen, {'factor': factor, 'axis': axis}


def generate_mesh(nx, ny):
    shape, dim = np.array((ny, nx)), 2
    node_numbers = np.arange(np.prod(shape + 1)).reshape(shape + 1, order='F')
    elem_num, dof = np.prod(shape), np.prod(shape + 1) * dim

    additions = np.array([0, 1, 2 * ny + 2, 2 * ny + 3, 2 * ny + 0, 2 * ny + 1, -2, -1])

    slicer = tuple(slice(None, -1) for _ in shape)
    c_vec = np.reshape(dim * node_numbers[slicer] + dim, (elem_num, 1), order='F')
    c_mat = c_vec + additions
    sI = np.hstack([np.arange(j, dim * 4) for j in range(dim * 4)])
    sII = np.hstack([np.tile(j, dim * 4 - j) for j in range(dim * 4)])
    iK, jK = c_mat[:, sI].T, c_mat[:, sII].T
    indexes = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]

    return {'dim': dim, 'shape': shape, 'dof': dof, 'node_numbers': node_numbers, 'elem_num': elem_num,
            'indexes': (indexes[:, 0], indexes[:, 1]), 'c_mat': c_mat}


def get_frozen(mesh, preserved_regions):
    frozen, frozen_mask = np.zeros(mesh['shape']), np.zeros(mesh['shape'], dtype=bool)
    for row in preserved_regions:
        start, end, name = [row[key] for key in ('S', 'E', 'Material')]
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1) for n, s, e
              in list(zip(mesh['shape'], *[(p[1], p[0], *p[2:]) for p in (start, end)]))]
        frozen_mask[nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]] = True
        frozen[nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]] = 1 if name.upper() == 'SOLID' else 0
    return frozen, frozen_mask


def get_bc(mesh, bc):
    fixed, dof = [], mesh['dof']
    force_vector = matrix(0.0, (dof, 1))
    for row in bc:
        start, end, displacement, force = [row[key] for key in ('S', 'E', 'D', 'F')]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e
              in list(zip(mesh['shape'], *[(p[1], p[0]) for p in (start, end)]))]
        nodes = (mesh['node_numbers'][nr[0][0]:nr[0][1], nr[1][0]:nr[1][1]]).flatten()

        for dim in range(mesh['dim']):
            if callable(force):
                for node in nodes:
                    coordinates = np.flip(np.argwhere(mesh['node_numbers'] == node)[0] / mesh['shape'])
                    force_vector[(mesh['dim'] * nodes + dim).tolist(), 0] = force(*coordinates)[dim]
            fixed.extend([] if np.isnan(displacement[dim]) else (mesh['dim'] * nodes + dim).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'fix_dofs': fixed, 'force_vector': force_vector}


def apply_symmetry(symmetry_axis, boundary_conditions, shape):
    if symmetry_axis is not None:
        bc = []
        shape = list(shape)

        axis = 0 if symmetry_axis.upper() == 'Y' else 1
        shape[axis] //= 2
        bc.append({'S': ((1, 0), (0, 1))[axis], 'E': (1, 1), 'D': ((0, nan), (nan, 0))[axis], 'F': 0})
        for row in boundary_conditions:
            start, end, d, force = [row[key] for key in ('S', 'E', 'D', 'F')]
            s, e = list(start), list(end)
            s[axis], e[axis] = min(2 * s[axis], 1), min(2 * e[axis], 1)
            if callable(force):
                force = F(force, axis).f
            if start[axis] <= 0.5 or end[axis] <= 0.5:
                bc.append({'S': s, 'E': e, 'D': d, 'F': force})
        return bc, shape, 2, int(not axis)
    else:
        return boundary_conditions, shape, 1, -1


class F:
    def __init__(self, force, axis):
        self.force, self.axis = force, axis

    def f(self, x, y):
        f = np.array(self.force(0.5 * x, 0.5 * y))
        f = f * (0.5 if (x, y)[self.axis] == 1 else 1)
        return f.tolist()
