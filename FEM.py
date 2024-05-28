import numpy as np
from cvxopt import spmatrix, cholmod
from scipy.sparse import csc_matrix


class FEM:

    def __init__(self, mesh, bc):
        self.shape, self.dof, self.indexes, self.dim = [mesh[k] for k in ['shape', 'dof', 'indexes', 'dim']]
        self.free, self.force = bc['free_dofs'], bc['force_vector']
        self.k, self.k_tri = Q4_stiffness()
        self.u = np.zeros(self.dof)

    def solve(self, e):
        ks = ((self.k_tri[np.newaxis]).T * (0.01 + e)).flatten(order='F')
        k = csc_matrix((ks, (self.indexes[0], self.indexes[1])),
                       shape=(self.dof, self.dof))[self.free, :][:, self.free].tocoo()
        k = spmatrix(k.data, k.row.astype(np.int32), k.col.astype(np.int32))
        u, b = np.zeros(self.dof), self.force[self.free]
        cholmod.linsolve(k, b)
        u[self.free] = np.array(b)[:, 0]
        self.u = u
        return u


def Q4_stiffness(nu=0.3):
    c1 = np.array([12, 3, -6, -3, -6, -3, 0, 3, 12, 3, 0, -3, -6, -3, -6, 12, -3, 0,
                   -3, -6, 3, 12, 3, -6, 3, -6, 12, 3, -6, -3, 12, 3, 0, 12, -3, 12])
    c2 = np.array([-4, 3, -2, 9, 2, -3, 4, -9, -4, -9, 4, -3, 2, 9, -2, -4, -3, 4,
                   9, 2, 3, -4, -9, -2, 3, 2, -4, 3, -2, 9, -4, -9, 4, -4, -3, -4])
    k_tri = 1 / (1 - nu ** 2) / 24 * (c1 + nu * c2)
    k = np.zeros((8, 8))
    k[np.triu_indices(8)] = k_tri
    k = k + k.T - np.diag(np.diag(k))
    return k, k_tri
