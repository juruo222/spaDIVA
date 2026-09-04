import numpy as np
import torch

from typing import Optional
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

__all__ = [
    "build_modality_graphs",
    "cal_spatial",
    "cluster",
    "clr_normalize_each_cell",
    "joint_cluster",
    "lsi",
]

def _run_mclust(Z_loc, num_cluster, random_seed):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector

    Z_loc = np.asarray(Z_loc, dtype=np.float64)
    if Z_loc.ndim == 1:
        Z_loc = Z_loc.reshape(-1, 1)
    importr("mclust")
    ro.r["set.seed"](random_seed)
    r_data = ro.r["matrix"](
        FloatVector(Z_loc.ravel(order="F")),
        nrow=Z_loc.shape[0],
        ncol=Z_loc.shape[1],
    )
    ro.globalenv["spadiva_mclust_data"] = r_data
    ro.globalenv["spadiva_mclust_g"] = num_cluster
    mclust_result = ro.r(
        'mclust::Mclust(spadiva_mclust_data, G=spadiva_mclust_g, modelNames="EEE")'
    )
    if mclust_result is None or "NULLType" in str(type(mclust_result)):
        raise RuntimeError("mclust returned NULL.")
    classification = mclust_result.rx2("classification")
    return np.asarray(classification)


def cal_spatial(points, k=6):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(points)
    _, indices = nbrs.kneighbors(points)

    edge_index_list = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            edge_index_list.append([neighbor, i])
    edge_index = torch.tensor(edge_index_list).t().contiguous()
    edge_index_sym = to_undirected(edge_index)

    return edge_index_sym


def _to_numpy_array(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    import scipy
    if scipy.sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _feature_graph(x, k=20, metric="minkowski"):
    x = _to_numpy_array(x)
    n_obs = x.shape[0]
    n_neighbors = min(k + 1, n_obs)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric=metric).fit(x)
    distances, indices = nbrs.kneighbors(x)
    edge_dict = {}
    for i in range(n_obs):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            if metric in {"correlation", "cosine"}:
                sim = max(0.0, 1.0 - float(dist))
            else:
                sim = 1.0 / (1.0 + float(dist))
            edge_dict[(int(j), int(i))] = max(edge_dict.get((int(j), int(i)), 0.0), sim)
            edge_dict[(int(i), int(j))] = max(edge_dict.get((int(i), int(j)), 0.0), sim)
    if not edge_dict:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    edges = np.array(list(edge_dict.keys()), dtype=np.int64).T
    weights = np.array(list(edge_dict.values()), dtype=np.float32)
    return torch.tensor(edges, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


def _combine_graphs(spatial_edge_index, feature_edge_index, feature_edge_weight, spatial_weight=1.0, feature_weight=0.0):
    if feature_weight == 0:
        return spatial_edge_index, None
    spatial_edges = _to_numpy_array(spatial_edge_index).astype(np.int64)
    feature_edges = _to_numpy_array(feature_edge_index).astype(np.int64)
    feature_weights = _to_numpy_array(feature_edge_weight).astype(np.float32)
    edge_weight_dict = {}
    for src, dst in spatial_edges.T:
        key = (int(src), int(dst))
        edge_weight_dict[key] = edge_weight_dict.get(key, 0.0) + float(spatial_weight)
    for (src, dst), w in zip(feature_edges.T, feature_weights):
        key = (int(src), int(dst))
        edge_weight_dict[key] = edge_weight_dict.get(key, 0.0) + float(feature_weight) * float(w)
    edges = np.array(list(edge_weight_dict.keys()), dtype=np.int64).T
    weights = np.array(list(edge_weight_dict.values()), dtype=np.float32)
    return torch.tensor(edges, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


def build_modality_graphs(x1, x2, edge_index, k_feature=20, spatial_weight=1.0, feature_weight=0.0, feature_metric="minkowski"):
    if feature_weight == 0:
        return edge_index, None, edge_index, None
    feature_edge_index1, feature_edge_weight1 = _feature_graph(x1, k=k_feature, metric=feature_metric)
    feature_edge_index2, feature_edge_weight2 = _feature_graph(x2, k=k_feature, metric=feature_metric)
    edge_index1, edge_weight1 = _combine_graphs(edge_index, feature_edge_index1, feature_edge_weight1, spatial_weight, feature_weight)
    edge_index2, edge_weight2 = _combine_graphs(edge_index, feature_edge_index2, feature_edge_weight2, spatial_weight, feature_weight)
    return edge_index1, edge_weight1, edge_index2, edge_weight2


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X)
    return adata


def lsi(
    adata,
    n_components: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs,
) -> None:
    """Run LSI analysis following the Seurat v3 approach."""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    from sklearn.preprocessing import Normalizer
    from sklearn.utils.extmath import randomized_svd

    X_norm = Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    return X_lsi[:, 1:]


def tfidf(X):
    """Run TF-IDF normalization following the Seurat v3 approach."""
    idf = X.shape[0] / X.sum(axis=0)
    import scipy

    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    tf = X / X.sum(axis=1, keepdims=True)
    return tf * idf


def cluster(
    Z_loc,
    num_cluster,
    spatial,
    title="",
    random_seed=42,
    s=50,
    show=True,
    return_labels=False,
    legend_loc="none",
    Y=None,
    cluster_method="mclust",
):
    import scanpy as sc
    import matplotlib.pyplot as plt

    z_adata = sc.AnnData(Z_loc)
    z_adata.obsm['spatial'] = spatial

    pred_z_mclust = None

    if cluster_method == "mclust":
        pred_z_mclust = _run_mclust(Z_loc, num_cluster, random_seed)
    elif cluster_method != "gmm":
        raise ValueError("cluster_method must be 'mclust' or 'gmm'.")

    if pred_z_mclust is None:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=num_cluster,
            covariance_type="tied",
            random_state=random_seed,
            n_init=10,
        )
        pred_z_mclust = gmm.fit_predict(np.asarray(Z_loc)) + 1

    labels = pred_z_mclust.astype(str)
    z_adata.obs['mclust_diag'] = labels

    if show:
        sc.set_figure_params(figsize=(3, 3))

        ax = sc.pl.embedding(
            z_adata, basis='spatial', color='mclust_diag', s=s, show=False, legend_loc=legend_loc,
        )
        ax.set_title("", fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    if Y is not None:
        if show is True:
            import umap
            umaper=umap.UMAP(random_state=random_seed)
            z_loc=umaper.fit_transform(Z_loc)
            for cell_type in np.sort(np.unique(Y)):
                idx = Y == cell_type
                plt.scatter(z_loc[idx,0],z_loc[idx,1],label=cell_type,s=4)
            plt.legend(loc="center left",bbox_to_anchor=(1, 0.5))
            plt.title(title+'_umap')
            plt.show()
        ARI = adjusted_rand_score(Y, pred_z_mclust)
        NMI = normalized_mutual_info_score(Y, pred_z_mclust)
        if return_labels:
            return ARI, NMI, labels
        else:
            return ARI, NMI

    if return_labels:
        return labels
    return None

def joint_cluster(z_adata, num_cluster, title, random_seed=42, show=True, Y1=None, Y2=None, Y3=None, cluster_method="mclust"):
    Z_loc = z_adata.X
    import scanpy as sc
    import matplotlib.pyplot as plt
    from matplotlib import cm

    pred_z_mclust = None

    if cluster_method == "mclust":
        pred_z_mclust = _run_mclust(Z_loc, num_cluster, random_seed)
    elif cluster_method != "gmm":
        raise ValueError("cluster_method must be 'mclust' or 'gmm'.")

    if pred_z_mclust is None:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=num_cluster,
            covariance_type="tied",
            random_state=random_seed,
            n_init=10,
        )
        pred_z_mclust = gmm.fit_predict(np.asarray(Z_loc)) + 1

    z_adata.obs["mclust_diag"] = pred_z_mclust.astype(str).astype(float).astype(int).astype("str")

    cluster_labels = z_adata.obs["mclust_diag"]
    cluster_slice1 = cluster_labels[z_adata.obs["batch"] == 0]
    cluster_slice2 = cluster_labels[z_adata.obs["batch"] == 1]
    cluster_slice3 = cluster_labels[z_adata.obs["batch"] == 2]
    have_labels = (Y1 is not None) and (Y2 is not None) and (Y3 is not None)
    if have_labels:
        ari1 = adjusted_rand_score(Y1, cluster_slice1)
        ari2 = adjusted_rand_score(Y2, cluster_slice2)
        ari3 = adjusted_rand_score(Y3, cluster_slice3)
        print(f"slice1 ari: {ari1}, slice2 ari: {ari2}, slice3 ari: {ari3}")

    if show is True:
        adata_slice1 = z_adata[z_adata.obs["batch"] == 0].copy()
        adata_slice2 = z_adata[z_adata.obs["batch"] == 1].copy()
        adata_slice3 = z_adata[z_adata.obs["batch"] == 2].copy()

        unique_clusters = sorted(cluster_labels.astype(int).unique())
        colors = cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        palette = {str(c): colors[i] for i, c in enumerate(unique_clusters)}

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

        with plt.rc_context({"axes.facecolor": "white"}):
            sc.pl.embedding(
                adata_slice1,
                basis="spatial",
                color="mclust_diag",
                palette=palette,
                title=f"{title} - Slice1",
                ax=ax1,
                show=False,
                s=50,
                legend_loc=None
            )

        with plt.rc_context({"axes.facecolor": "white"}):
            sc.pl.embedding(
                adata_slice2,
                basis="spatial",
                color="mclust_diag",
                palette=palette,
                title=f"{title} - Slice2",
                ax=ax2,
                show=False,
                s=50,
                legend_loc=None
            )

        with plt.rc_context({"axes.facecolor": "white"}):
            sc.pl.embedding(
                adata_slice3,
                basis="spatial",
                color="mclust_diag",
                palette=palette,
                title=f"{title} - Slice3",
                ax=ax3,
                show=False,
                s=50,
                legend_loc=None
            )

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markersize=10)
            for i in range(len(unique_clusters))
        ]
        fig.legend(
            handles,
            [f"Cluster {c}" for c in unique_clusters],
            loc="lower center",
            ncol=min(5, len(unique_clusters)),
            bbox_to_anchor=(0.5, -0.30)
        )

        plt.tight_layout()
        plt.show()

    if have_labels:
        return ari1, ari2, ari3
    return None

def n_n(ds1, ds2, names1, names2, knn=50, metric_p=2):
    n_neighbors = min(knn, ds2.shape[0])
    nn_ = NearestNeighbors(n_neighbors=n_neighbors, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def n_n_hnsw(ds1, ds2, names1, names2, knn=50):
    import hnswlib

    ds1 = np.ascontiguousarray(ds1, dtype=np.float32)
    ds2 = np.ascontiguousarray(ds2, dtype=np.float32)
    n_neighbors = min(knn, ds2.shape[0])
    if n_neighbors <= 0:
        return set()

    index = hnswlib.Index(space="l2", dim=ds2.shape[1])
    index.init_index(
        max_elements=ds2.shape[0],
        ef_construction=max(100, 2 * n_neighbors),
        M=16,
    )
    index.add_items(ds2, np.arange(ds2.shape[0], dtype=np.int64))
    index.set_ef(max(50, 2 * n_neighbors))
    indices, _ = index.knn_query(ds1, k=n_neighbors)

    match = set()
    for a, neighbors in zip(range(ds1.shape[0]), indices):
        for b_i in neighbors:
            match.add((names1[a], names2[int(b_i)]))

    return match


def zscore_np(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < eps] = 1.0
    return (X - mu) / sd


def mnn(ds1, ds2, names1, names2, knn=20, approx=True):
    if approx:
        match1 = n_n_hnsw(ds1, ds2, names1, names2, knn=knn)
        match2 = n_n_hnsw(ds2, ds1, names2, names1, knn=knn)
    else:
        match1 = n_n(ds1, ds2, names1, names2, knn=knn)
        match2 = n_n(ds2, ds1, names2, names1, knn=knn)
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def update_mnn(
    embedding1,
    embedding2,
    knn=50,
    approx=True,
    standardize=True,
):

    if isinstance(embedding1, torch.Tensor):
        ds1 = embedding1.detach().cpu().numpy()
    else:
        ds1 = np.asarray(embedding1)

    if isinstance(embedding2, torch.Tensor):
        ds2 = embedding2.detach().cpu().numpy()
    else:
        ds2 = np.asarray(embedding2)

    ds1 = np.asarray(ds1, dtype=np.float32)
    ds2 = np.asarray(ds2, dtype=np.float32)

    if standardize:
        ds1 = zscore_np(ds1)
        ds2 = zscore_np(ds2)

    names1 = list(range(ds1.shape[0]))
    names2 = list(range(ds2.shape[0]))

    match = mnn(
        ds1,
        ds2,
        names1,
        names2,
        knn=knn,
        approx=approx,
    )

    return match


def compute_MNN_loss(
    z1,
    z2,
    match,
    tau=0.5,
    max_mnn_pairs=None,
):
    """Compute triplet loss from mutual nearest-neighbor pairs."""
    if not match:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

    if z1.size(0) < 2:
        raise ValueError(
            "z1 must contain at least 2 samples for nearest-neighbor negative sampling."
        )

    match_list = sorted(list(match))

    if max_mnn_pairs is not None and len(match_list) > max_mnn_pairs:
        perm = torch.randperm(len(match_list), device=z1.device)[:max_mnn_pairs]
        match_list = [match_list[int(i)] for i in perm.detach().cpu().numpy()]

    a_indices, p_indices = zip(*match_list)

    a_tensor = torch.as_tensor(
        a_indices,
        device=z1.device,
        dtype=torch.long,
    )

    p_tensor = torch.as_tensor(
        p_indices,
        device=z2.device,
        dtype=torch.long,
    )

    anchors = z1[a_tensor]
    positives = z2[p_tensor]

    n_pairs = a_tensor.numel()

    nn_indices = torch.empty(
        n_pairs,
        device=z1.device,
        dtype=torch.long,
    )

    nn_chunk_size = 1024

    with torch.no_grad():
        z1_detached = z1.detach()

        for start in range(0, n_pairs, nn_chunk_size):
            end = min(start + nn_chunk_size, n_pairs)

            anchor_idx_chunk = a_tensor[start:end]
            anchor_chunk = z1_detached[anchor_idx_chunk]

            dist = torch.cdist(anchor_chunk, z1_detached, p=2) ** 2

            row_ids = torch.arange(
                end - start,
                device=z1.device,
            )

            dist[row_ids, anchor_idx_chunk] = float("inf")

            nn_indices[start:end] = torch.argmin(dist, dim=1)

    negatives = z1[nn_indices]

    triplet_loss = torch.nn.TripletMarginLoss(
        margin=tau,
        p=2,
        reduction="mean",
    )

    loss = triplet_loss(
        anchors,
        positives,
        negatives,
    )

    return loss


def reverse_matches(original_matches):
    """Reverse matched index pairs from (z1_idx, z2_idx) to (z2_idx, z1_idx)."""
    return {(p_idx, a_idx) for a_idx, p_idx in original_matches}
