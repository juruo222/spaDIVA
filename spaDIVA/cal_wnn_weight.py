import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

__all__ = ["cal_weight"]


def dis(A, B):
    return np.sqrt(np.sum((A - B) ** 2, axis=1))


def cal_weight(z1, z2, k=20):
    """Compute WNN-style weights for two latent representations."""

    if isinstance(z1, torch.Tensor):
        z1 = z1.cpu().detach().numpy()
    if isinstance(z2, torch.Tensor):
        z2 = z2.cpu().detach().numpy()

    nbrs1 = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(z1)
    nbrs2 = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(z2)

    _, indices1 = nbrs1.kneighbors(z1)
    _, indices2 = nbrs2.kneighbors(z2)

    neighbors1 = indices1[:, 1:k + 1]
    neighbors2 = indices2[:, 1:k + 1]

    z1_in_z1 = z1[neighbors1].mean(axis=1)
    z2_in_z1 = z2[neighbors1].mean(axis=1)
    z1_in_z2 = z1[neighbors2].mean(axis=1)
    z2_in_z2 = z2[neighbors2].mean(axis=1)

    band_width1 = dis(z1, z1[indices1[:, -1]])
    band_width2 = dis(z2, z2[indices2[:, -1]])
    first_neighbor_in_1 = dis(z1, z1[indices1[:, 1]])
    first_neighbor_in_2 = dis(z2, z2[indices2[:, 1]])

    arr1_in_1 = dis(z1, z1_in_z1) - first_neighbor_in_1
    arr1_in_2 = dis(z1, z1_in_z2) - first_neighbor_in_1

    arr2_in_2 = dis(z2, z2_in_z2) - first_neighbor_in_2
    arr2_in_1 = dis(z2, z2_in_z1) - first_neighbor_in_2

    arr1_in_1[arr1_in_1 < 0] = 0
    arr2_in_2[arr2_in_2 < 0] = 0
    arr1_in_2[arr1_in_2 < 0] = 0
    arr2_in_1[arr2_in_1 < 0] = 0

    denom1 = band_width1 - first_neighbor_in_1
    denom2 = band_width2 - first_neighbor_in_2

    tiny = 1e-8
    denom1 = np.maximum(denom1, tiny)
    denom2 = np.maximum(denom2, tiny)

    aff1_in_1 = np.exp(-arr1_in_1 / denom1)
    aff2_in_2 = np.exp(-arr2_in_2 / denom2)
    aff1_in_2 = np.exp(-arr1_in_2 / denom1)
    aff2_in_1 = np.exp(-arr2_in_1 / denom2)

    eps = 1e-4
    c1 = aff1_in_1 / (aff1_in_2 + eps)
    c2 = aff2_in_2 / (aff2_in_1 + eps)

    max_values = np.maximum(c1, c2)

    c1 = np.exp(c1 - max_values)
    c2 = np.exp(c2 - max_values)

    a1 = c1 / (c1 + c2)
    a2 = c2 / (c1 + c2)

    return a1, a2
