import numpy as np

from spaDIVA.cal_wnn_weight import cal_weight

__all__ = [
    "collect_spadiva_outputs",
    "fuse_shared_latent",
    "make_integrated_embedding",
]


def fuse_shared_latent(Z1, Z2, k=20, return_weights=False):
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    if Z1.shape != Z2.shape:
        raise ValueError(f"Z1 and Z2 must have the same shape, got {Z1.shape} and {Z2.shape}.")
    w1, w2 = cal_weight(Z1, Z2, k=k)
    Z = w1.reshape(-1, 1) * Z1 + w2.reshape(-1, 1) * Z2
    if return_weights:
        return Z, w1, w2
    return Z


def make_integrated_embedding(Z, W1, W2):
    return np.concatenate((np.asarray(W1), np.asarray(Z), np.asarray(W2)), axis=1)


def collect_spadiva_outputs(
    Z,
    W1,
    W2,
    spatial,
    obs_names=None,
    modality_names=("omics1", "omics2"),
    Z_poe=None,
    Z1=None,
    Z2=None,
    X1_hat=None,
    X2_hat=None,
    model_seed=None,
    poe_sample_seed=None,
):
    import scanpy as sc

    if len(modality_names) != 2:
        raise ValueError("modality_names must contain exactly two names.")

    Z = np.asarray(Z)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Z_W = make_integrated_embedding(Z, W1, W2)

    adata = sc.AnnData(X=Z)
    adata.obsm["Z"] = Z
    adata.obsm["Z_W"] = Z_W
    adata.obsm[f"W_{modality_names[0]}"] = W1
    adata.obsm[f"W_{modality_names[1]}"] = W2
    adata.obsm["spatial"] = np.asarray(spatial)

    if obs_names is not None:
        adata.obs_names = obs_names.copy() if hasattr(obs_names, "copy") else list(obs_names)
    if Z_poe is not None:
        adata.obsm["Z_poe"] = np.asarray(Z_poe)
    if model_seed is not None:
        adata.uns["model_seed"] = int(model_seed)
    if poe_sample_seed is not None:
        adata.uns["poe_sample_seed"] = int(poe_sample_seed)
    if Z1 is not None:
        adata.obsm[f"Z_{modality_names[0]}"] = np.asarray(Z1)
    if Z2 is not None:
        adata.obsm[f"Z_{modality_names[1]}"] = np.asarray(Z2)
    if X1_hat is not None:
        adata.obsm[f"X_hat_{modality_names[0]}"] = np.asarray(X1_hat)
    if X2_hat is not None:
        adata.obsm[f"X_hat_{modality_names[1]}"] = np.asarray(X2_hat)

    return adata
