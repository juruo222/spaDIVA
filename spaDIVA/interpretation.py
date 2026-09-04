import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

__all__ = [
    "add_unique_factor_ratio",
    "correlate_metrics",
    "evaluate_unique_factor_ratio",
    "featurewise_pearson",
    "featurewise_r2",
    "filter_features",
    "prepare_feature_translation_inputs",
    "ranked_rolling_mean",
    "repeated_subsampling_unique_factor_ratio",
    "ridge_predict",
    "select_shared_latent",
    "summarize_extreme_groups",
]

_POE_LATENT_KEYS = ("Z_poe", "Z_PoE")
_WNN_LATENT_KEYS = ("Z", "Z_WNN")


def _as_array(x, dtype=np.float64):
    import scipy.sparse

    if scipy.sparse.issparse(x):
        x = x.toarray()
    elif hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=dtype)


def _check_same_shape(a, b, a_name="a", b_name="b"):
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same shape, got {a.shape} and {b.shape}.")


def _paired_feature_arrays(y_true, y_pred):
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be one- or two-dimensional.")
    _check_same_shape(y_true, y_pred, "y_true", "y_pred")

    paired = np.isfinite(y_true) & np.isfinite(y_pred)
    counts = paired.sum(axis=0)
    return y_true, y_pred, paired, counts


def _center_paired_values(values, paired, counts):
    totals = np.sum(np.where(paired, values, 0.0), axis=0)
    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    centered = np.zeros_like(values)
    np.subtract(values, means, out=centered, where=paired)
    return centered


def featurewise_pearson(y_true, y_pred, eps=1e-12):
    """Compute Pearson correlation from finite observation pairs per feature.

    Features with fewer than two pairs or a constant vector return NaN.
    A one-dimensional input is treated as a single feature.
    """
    y_true, y_pred, paired, counts = _paired_feature_arrays(y_true, y_pred)
    y_true_centered = _center_paired_values(y_true, paired, counts)
    y_pred_centered = _center_paired_values(y_pred, paired, counts)

    numerator = np.sum(y_true_centered * y_pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(y_true_centered ** 2, axis=0)
        * np.sum(y_pred_centered ** 2, axis=0)
    )

    corr = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = (counts >= 2) & (denominator > eps)
    corr[valid] = numerator[valid] / denominator[valid]
    return corr


def featurewise_r2(y_true, y_pred, eps=1e-12):
    """Compute held-out R-squared from finite pairs for each feature.

    Features with fewer than two pairs or constant true values return NaN.
    Negative R-squared values are retained.
    """
    y_true, y_pred, paired, counts = _paired_feature_arrays(y_true, y_pred)
    residual = np.zeros_like(y_true)
    np.subtract(y_true, y_pred, out=residual, where=paired)
    sse = np.sum(residual ** 2, axis=0)
    y_centered = _center_paired_values(y_true, paired, counts)
    sst = np.sum(y_centered ** 2, axis=0)

    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = (counts >= 2) & (sst > eps)
    r2[valid] = 1.0 - sse[valid] / sst[valid]
    return r2


def ridge_predict(X_train, Y_train, X_test, alpha=1.0, solver="auto", fit_intercept=True):
    """Fit Ridge while preserving the target's one- or two-dimensional shape."""
    Y_train = _as_array(Y_train)
    model = Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept)
    model.fit(_as_array(X_train), Y_train)
    prediction = model.predict(_as_array(X_test))
    if Y_train.ndim == 2 and prediction.ndim == 1:
        prediction = prediction[:, None]
    return prediction


def add_unique_factor_ratio(
    df,
    r2_z_col="r2_z",
    r2_zw_col="r2_zw",
    r2_threshold=0.001,
    eps=1e-12,
    clip_gain=True,
):
    out = df.copy()
    r2_z = out[r2_z_col].to_numpy(dtype=np.float64)
    r2_zw = out[r2_zw_col].to_numpy(dtype=np.float64)

    delta_r2 = r2_zw - r2_z
    unique_factor_gain = np.maximum(delta_r2, 0.0) if clip_gain else delta_r2

    unique_factor_ratio = np.full_like(unique_factor_gain, np.nan, dtype=np.float64)
    valid = np.isfinite(r2_z) & np.isfinite(r2_zw) & (r2_zw > r2_threshold)
    unique_factor_ratio[valid] = unique_factor_gain[valid] / (r2_zw[valid] + eps)

    out["delta_r2"] = delta_r2
    out["unique_factor_gain"] = unique_factor_gain
    out["unique_factor_ratio"] = unique_factor_ratio
    out["r2_zw_min_for_ratio"] = r2_threshold
    return out


def filter_features(Y, min_nonzero_frac=0.0, min_variance=0.0, feature_names=None):
    Y = _as_array(Y)
    nonzero_frac = np.mean(Y != 0, axis=0)
    variance = np.nanvar(Y, axis=0)
    mask = (nonzero_frac >= min_nonzero_frac) & (variance > min_variance)

    stats = pd.DataFrame(
        {
            "feature_idx": np.arange(Y.shape[1]),
            "nonzero_frac": nonzero_frac,
            "variance": variance,
            "eligible": mask,
        }
    )
    if feature_names is not None:
        stats.insert(0, "feature_name", np.asarray(feature_names).astype(str))
    return mask, stats


def _load_anndata(x):
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        import scanpy as sc

        x = sc.read_h5ad(x)
    x.obs_names = x.obs_names.astype(str)
    x.var_names_make_unique()
    x.var_names = x.var_names.astype(str)
    return x


def _resolve_shared_latent_key(latent_adata, use="Z_poe"):
    if use is None:
        use = "Z_poe"
    if not isinstance(use, str):
        raise TypeError("use must be 'Z_poe' or 'Z'.")

    normalized = use.strip().lower()
    canonical = {"z_poe": "Z_poe", "z": "Z", "z_wnn": "Z"}
    if normalized not in canonical:
        raise ValueError("use must be 'Z_poe' or 'Z'.")

    requested = canonical[normalized]
    preferred = _POE_LATENT_KEYS if requested == "Z_poe" else _WNN_LATENT_KEYS
    for key in preferred:
        if key in latent_adata.obsm:
            return key

    if requested == "Z":
        raise KeyError('Neither obsm["Z"] nor obsm["Z_WNN"] was found in latent_adata.')

    for key in _WNN_LATENT_KEYS:
        if key in latent_adata.obsm:
            warnings.warn(
                'Neither obsm["Z_poe"] nor obsm["Z_PoE"] is available; '
                f'using obsm["{key}"] instead.',
                RuntimeWarning,
                stacklevel=3,
            )
            return key
    raise KeyError(
        'No shared latent found in obsm; expected "Z_poe" (alias "Z_PoE") '
        'or "Z" (alias "Z_WNN").'
    )


def select_shared_latent(latent_adata, use="Z_poe"):
    """Return the selected shared latent matrix and the actual AnnData key.

    ``Z_poe`` is preferred by default for unique factor ratio analysis. If it
    is unavailable, the function falls back to the WNN shared representation
    stored in ``Z``. ``Z_PoE`` and ``Z_WNN`` are accepted as aliases in both
    ``obsm`` and ``use``. Within each representation, the standard key takes
    precedence over its alias. Passing ``use="Z"`` selects only WNN, never PoE.
    Existing keys and arrays are not renamed or resampled.
    """
    latent_adata = _load_anndata(latent_adata)
    z_source = _resolve_shared_latent_key(latent_adata, use=use)
    return np.asarray(latent_adata.obsm[z_source], dtype=np.float32), z_source


def _validate_split_names(values, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of observation names.")
    if pd.isna(values).any():
        raise ValueError(f"{name} must not contain missing observation names.")
    names = pd.Index(values.astype(str))
    if len(names) < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if (names == "").any():
        raise ValueError(f"{name} must not contain empty observation names.")
    if not names.is_unique:
        raise ValueError(f"{name} must not contain duplicate observation names.")
    return names


def _validate_split_indices(values, name, n_obs):
    indices = np.asarray(values)
    contains_boolean = isinstance(values, (list, tuple)) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    )
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or contains_boolean:
        raise ValueError(f"{name} must be a one-dimensional integer index array.")
    if np.any(indices < 0) or np.any(indices >= n_obs):
        raise ValueError(f"{name} contains indices outside feature_adata's observation range.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate indices.")
    return indices


def _require_observations(names, available, description):
    missing = names[~names.isin(available)]
    if len(missing):
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"{description} is missing {len(missing)} required observation(s): {examples}."
        )


def _read_split_names(split, feature_adata):
    if isinstance(split, (str, bytes)) or hasattr(split, "__fspath__"):
        loaded = np.load(split, allow_pickle=True)
        try:
            return _read_split_names(loaded, feature_adata)
        finally:
            if hasattr(loaded, "close"):
                loaded.close()

    if not hasattr(split, "keys"):
        raise TypeError("split must be a mapping or an NPZ file.")
    files = set(split.keys())
    name_keys = {"train_names", "test_names"}
    if name_keys & files:
        if not name_keys.issubset(files):
            raise KeyError("split must contain both train_names and test_names.")
        train_names = _validate_split_names(split["train_names"], "train_names")
        test_names = _validate_split_names(split["test_names"], "test_names")
    else:
        if not {"idx_train", "idx_test"}.issubset(files):
            raise KeyError("split must contain train_names/test_names or idx_train/idx_test.")
        idx_train = _validate_split_indices(split["idx_train"], "idx_train", feature_adata.n_obs)
        idx_test = _validate_split_indices(split["idx_test"], "idx_test", feature_adata.n_obs)
        train_names = _validate_split_names(feature_adata.obs_names[idx_train], "train_names")
        test_names = _validate_split_names(feature_adata.obs_names[idx_test], "test_names")

    if train_names.isin(test_names).any():
        raise ValueError("Training and test observations must not overlap.")
    _require_observations(train_names, feature_adata.obs_names, "feature_adata for training")
    _require_observations(test_names, feature_adata.obs_names, "feature_adata for testing")
    return train_names, test_names


def prepare_feature_translation_inputs(
    latent_adata,
    feature_adata,
    external_true_adata,
    external_pred_adata,
    split,
    use="Z_poe",
    w_key="W_ADT",
    min_nonzero_frac=0.0,
    min_variance=0.0,
    max_features=None,
):
    """Align translation inputs without changing the declared train/test split.

    Integer split indices refer to the original row order of feature_adata,
    before alignment to latent_adata. Named splits take precedence when both
    formats are provided. All requested observations must be present: train
    and test in feature/latent data, and test in both external matrices.
    Unused observations may differ between inputs. Missing requested names,
    duplicates, invalid indices, and overlapping splits raise an error.
    """
    latent_adata = _load_anndata(latent_adata)
    feature_adata = _load_anndata(feature_adata)
    external_true_adata = _load_anndata(external_true_adata)
    external_pred_adata = _load_anndata(external_pred_adata)

    for name, adata in (
        ("latent_adata", latent_adata),
        ("feature_adata", feature_adata),
        ("external_true_adata", external_true_adata),
        ("external_pred_adata", external_pred_adata),
    ):
        if not adata.obs_names.is_unique:
            raise ValueError(f"{name}.obs_names must be unique; duplicate observations found.")

    train_names, test_names = _read_split_names(split, feature_adata)
    _require_observations(train_names, latent_adata.obs_names, "latent_adata for training")
    _require_observations(test_names, latent_adata.obs_names, "latent_adata for testing")
    _require_observations(test_names, external_true_adata.obs_names, "external_true_adata")
    _require_observations(test_names, external_pred_adata.obs_names, "external_pred_adata")

    z_source = _resolve_shared_latent_key(latent_adata, use=use)
    if w_key not in latent_adata.obsm:
        raise KeyError(f'obsm["{w_key}"] not found in latent_adata.')

    train_latent = latent_adata[train_names].copy()
    test_latent = latent_adata[test_names].copy()
    train_features = feature_adata[train_names].copy()
    external_true = external_true_adata[test_names].copy()
    external_pred = external_pred_adata[test_names].copy()

    common_features = (
        train_features.var_names
        .intersection(external_true.var_names)
        .intersection(external_pred.var_names)
    )
    if len(common_features) == 0:
        raise ValueError("No common features in feature_adata and the external matrices.")
    train_features = train_features[:, common_features].copy()
    external_true = external_true[:, common_features].copy()
    external_pred = external_pred[:, common_features].copy()
    external_true = external_true[:, train_features.var_names].copy()
    external_pred = external_pred[:, train_features.var_names].copy()

    Y_train = _as_array(train_features.X, dtype=np.float32)
    Y_test_true = _as_array(external_true.X, dtype=np.float32)
    Y_test_pred = _as_array(external_pred.X, dtype=np.float32)
    feature_names = np.asarray(external_true.var_names).astype(str)

    eligible_mask, feature_stats = filter_features(
        Y_train,
        min_nonzero_frac=min_nonzero_frac,
        min_variance=min_variance,
        feature_names=feature_names,
    )
    feature_indices = np.where(eligible_mask)[0]
    if max_features is not None:
        if (
            isinstance(max_features, (bool, np.bool_))
            or not isinstance(max_features, (int, np.integer))
            or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None.")
        feature_indices = feature_indices[:max_features]

    return {
        "Z_train": np.asarray(train_latent.obsm[z_source], dtype=np.float32),
        "W_train": np.asarray(train_latent.obsm[w_key], dtype=np.float32),
        "Y_train": Y_train,
        "Z_test": np.asarray(test_latent.obsm[z_source], dtype=np.float32),
        "W_test": np.asarray(test_latent.obsm[w_key], dtype=np.float32),
        "Y_test_true": Y_test_true,
        "external_pred": Y_test_pred,
        "feature_names": feature_names,
        "feature_indices": feature_indices,
        "feature_stats": feature_stats,
        "train_names": train_names,
        "test_names": test_names,
        "z_source": z_source,
        "model_seed": latent_adata.uns.get("model_seed"),
        "poe_sample_seed": (
            latent_adata.uns.get("poe_sample_seed") if z_source in _POE_LATENT_KEYS else None
        ),
    }


def _select_features(Y, feature_indices):
    Y = _as_array(Y)
    if feature_indices is None:
        return Y
    return Y[:, np.asarray(feature_indices, dtype=int)]


def _select_external_prediction(Y_pred, feature_indices, n_selected):
    Y_pred = _as_array(Y_pred)
    if feature_indices is None:
        return Y_pred
    if Y_pred.shape[1] == n_selected:
        return Y_pred
    return Y_pred[:, np.asarray(feature_indices, dtype=int)]


def _feature_names(feature_names, feature_indices, n_features):
    if feature_names is None:
        if feature_indices is None:
            return np.array([f"feature_{i}" for i in range(n_features)], dtype=object)
        return np.asarray([f"feature_{i}" for i in feature_indices], dtype=object)

    names = np.asarray(feature_names).astype(str)
    if feature_indices is not None and len(names) != n_features:
        names = names[np.asarray(feature_indices, dtype=int)]
    return names


def evaluate_unique_factor_ratio(
    Z_train,
    W_train,
    Y_train,
    Z_test,
    W_test,
    Y_test_true,
    external_pred=None,
    feature_names=None,
    feature_indices=None,
    external_metric_name="external_feature_pearson",
    ridge_alpha=1.0,
    ridge_solver="auto",
    r2_threshold=0.001,
    eps=1e-12,
    clip_gain=True,
    w_key="W",
    z_source=None,
    model_seed=None,
    poe_sample_seed=None,
):
    Z_train = _as_array(Z_train)
    W_train = _as_array(W_train)
    Z_test = _as_array(Z_test)
    W_test = _as_array(W_test)
    Y_train = _select_features(Y_train, feature_indices)
    Y_test_true = _select_features(Y_test_true, feature_indices)

    if Z_train.shape[0] != W_train.shape[0] or Z_test.shape[0] != W_test.shape[0]:
        raise ValueError("Z and W must have matching sample counts within train and test sets.")
    if Y_train.shape[0] != Z_train.shape[0] or Y_test_true.shape[0] != Z_test.shape[0]:
        raise ValueError("Y matrices must have the same sample counts as the corresponding latent matrices.")

    ZW_train = np.concatenate([Z_train, W_train], axis=1)
    ZW_test = np.concatenate([Z_test, W_test], axis=1)

    Yhat_z = ridge_predict(Z_train, Y_train, Z_test, alpha=ridge_alpha, solver=ridge_solver)
    Yhat_w = ridge_predict(W_train, Y_train, W_test, alpha=ridge_alpha, solver=ridge_solver)
    Yhat_zw = ridge_predict(ZW_train, Y_train, ZW_test, alpha=ridge_alpha, solver=ridge_solver)

    n_features = Y_test_true.shape[1]
    if feature_indices is None:
        feature_idx = np.arange(n_features)
    else:
        feature_idx = np.asarray(feature_indices, dtype=int)

    result = pd.DataFrame(
        {
            "feature_name": _feature_names(feature_names, feature_indices, n_features),
            "feature_idx": feature_idx,
            "r2_z": featurewise_r2(Y_test_true, Yhat_z, eps=eps),
            "r2_w": featurewise_r2(Y_test_true, Yhat_w, eps=eps),
            "r2_zw": featurewise_r2(Y_test_true, Yhat_zw, eps=eps),
            "w_key": w_key,
        }
    )

    if z_source is not None:
        result["z_source"] = str(z_source)
    if model_seed is not None:
        result["model_seed"] = int(model_seed)
    if poe_sample_seed is not None:
        result["poe_sample_seed"] = int(poe_sample_seed)

    if external_pred is not None:
        external_pred = _select_external_prediction(external_pred, feature_indices, n_features)
        result[external_metric_name] = featurewise_pearson(Y_test_true, external_pred, eps=eps)

    result["nonzero_frac_train"] = np.mean(Y_train != 0, axis=0)
    result["variance_train"] = np.nanvar(Y_train, axis=0)

    return add_unique_factor_ratio(
        result,
        r2_threshold=r2_threshold,
        eps=eps,
        clip_gain=clip_gain,
    )


def correlate_metrics(x, y):
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)

    out = {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "n_valid": int(mask.sum())}
    if mask.sum() < 3:
        return out
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return out

    pr, pp = pearsonr(x[mask], y[mask])
    sr, sp = spearmanr(x[mask], y[mask])
    out.update(
        {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
        }
    )
    return out


def ranked_rolling_mean(df, x_col="unique_factor_ratio", y_col="external_feature_pearson", window=None, min_periods=1):
    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df.sort_values(x_col).reset_index(drop=True)
    if window is None:
        window = max(5, int(round(len(plot_df) * 0.10)))
    plot_df[f"{y_col}_rolling"] = (
        plot_df[y_col]
        .rolling(window=window, center=True, min_periods=min_periods)
        .mean()
    )
    return plot_df


def summarize_extreme_groups(
    df,
    x_col="unique_factor_ratio",
    y_col="external_feature_pearson",
    fraction=0.2,
    alternative="greater",
):
    from scipy.stats import mannwhitneyu

    if fraction < 0:
        raise ValueError("fraction must be non-negative.")

    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df.sort_values(x_col).reset_index(drop=True)
    n_each = max(1, int(round(len(plot_df) * fraction)))
    low = plot_df.head(n_each)
    high = plot_df.tail(n_each)

    low_y = low[y_col].to_numpy(dtype=np.float64)
    high_y = high[y_col].to_numpy(dtype=np.float64)
    stat, p_value = mannwhitneyu(low_y, high_y, alternative=alternative)

    return {
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "low_mean": float(np.nanmean(low_y)),
        "high_mean": float(np.nanmean(high_y)),
        "low_median": float(np.nanmedian(low_y)),
        "high_median": float(np.nanmedian(high_y)),
        "delta_low_minus_high": float(np.nanmean(low_y) - np.nanmean(high_y)),
        "mannwhitney_u": float(stat),
        "mannwhitney_p": float(p_value),
        "mannwhitney_alternative": alternative,
    }


def repeated_subsampling_unique_factor_ratio(
    Z_train,
    W_train,
    Y_train,
    Z_test,
    W_test,
    Y_test_true,
    external_pred,
    feature_names=None,
    eligible_indices=None,
    n_features=None,
    seeds=(2021, 2022, 2023, 2024, 2025),
    external_metric_name="external_feature_pearson",
    **kwargs,
):
    Y_train_all = _as_array(Y_train)
    n_total_features = Y_train_all.shape[1]

    if eligible_indices is None:
        eligible_indices = np.arange(n_total_features)
    eligible_indices = np.asarray(eligible_indices, dtype=int)

    if n_features is None:
        n_features = len(eligible_indices)
    if len(eligible_indices) < n_features:
        raise ValueError("eligible_indices contains fewer features than n_features.")

    all_results = []
    all_corr = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        sampled_idx = np.sort(rng.choice(eligible_indices, size=n_features, replace=False))
        result = evaluate_unique_factor_ratio(
            Z_train=Z_train,
            W_train=W_train,
            Y_train=Y_train,
            Z_test=Z_test,
            W_test=W_test,
            Y_test_true=Y_test_true,
            external_pred=external_pred,
            feature_names=feature_names,
            feature_indices=sampled_idx,
            external_metric_name=external_metric_name,
            **kwargs,
        )
        result["sample_seed"] = seed
        corr = correlate_metrics(result["unique_factor_ratio"], result[external_metric_name])
        corr.update({"sample_seed": seed, "metric": "unique_factor_ratio", "external_metric": external_metric_name})
        all_results.append(result)
        all_corr.append(corr)

    return pd.concat(all_results, axis=0, ignore_index=True), pd.DataFrame(all_corr)
