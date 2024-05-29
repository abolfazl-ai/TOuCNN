import numpy as np
import torch
from cvxopt import spmatrix, cholmod
from scipy.sparse import csc_matrix


class FEM:

    def __init__(self, mesh, bc):
        self.shape, self.dof, self.indexes, self.cmat = [mesh[k] for k in ['shape', 'dof', 'indexes', 'c_mat']]
        self.free, self.force = bc['free_dofs'], bc['force_vector']
        self.k, self.k_tri = Q4_stiffness()
        self.u = np.zeros(self.dof)

    def solve(self, e):
        ks = ((self.k_tri[np.newaxis]).T * (0.01 + e)).flatten(order='F')
        k = csc_matrix((ks, (self.indexes[0], self.indexes[1])),
                       shape=(self.dof, self.dof))[self.free, :][:, self.free].tocoo()
        k = spmatrix(k.data, k.row.astype(np.int32), k.col.astype(np.int32))
        f = self.force[self.free]
        cholmod.linsolve(k, f)
        self.u[self.free] = np.array(f)[:, 0]
        je = ((self.u[self.cmat] @ self.k) * self.u[self.cmat]).sum(1)
        return self.u, je.reshape(self.shape, order='F')

    def solve2(self, e):
        values = ((self.k_tri[np.newaxis]).T * (0.01 + e)).flatten(order='F')

        # Create indices for the sparse matrix in PyTorch
        indices = torch.tensor(self.indexes, dtype=torch.int64)
        values = torch.tensor(ks, dtype=torch.float32)

        # Create the sparse matrix
        k = torch.sparse_coo_tensor(indices, values, (self.dof, self.dof), dtype=torch.float32)

        free_indices = torch.tensor(self.free, dtype=torch.int64)
        k = k.index_select(0, free_indices).index_select(1, free_indices)

        # Convert to COO format
        k = k.coalesce()

        # Convert to scipy format for linsolve
        from scipy.sparse import coo_matrix
        k_scipy = coo_matrix((k.values().numpy(), (k.indices()[0].numpy(), k.indices()[1].numpy())),
                             shape=(len(self.free), len(self.free)))

        f = self.force[self.free]
        f_torch = torch.tensor(f, dtype=torch.float32)
        cholmod.linsolve(k_scipy, f_torch.numpy())

        # Update displacements
        self.u[self.free] = f_torch.numpy()[:, 0]

        # Compute energy
        u_torch = torch.tensor(self.u, dtype=torch.float32)
        je = ((u_torch[self.cmat] @ torch.tensor(self.k, dtype=torch.float32)) * u_torch[self.cmat]).sum(1)

        return self.u, je.numpy().reshape(self.shape, order='F')


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
