from spaDIVA.model import VariationalAutoEncoder
from spaDIVA.cal_wnn_weight import cal_weight
from contextlib import nullcontext
from functools import lru_cache
import warnings
import torch

__all__ = [
    "infer_latents",
    "joint_train_spadiva",
    "train_spadiva",
]


@lru_cache(maxsize=1)
def _cuda_is_usable():
    """Return whether this process can execute a real CUDA kernel.

    ``torch.cuda.get_arch_list()`` describes the architectures compiled into
    the wheel, but the absence of an exact ``sm_*`` entry is not conclusive:
    a newer GPU may still run embedded PTX after driver JIT compilation.
    Therefore, use the architecture list only to explain a potentially slow
    first call and decide availability from an actual CUDA computation.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        compute = f"compute_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if supported and arch not in supported and compute not in supported:
            warnings.warn(
                f"The current PyTorch wheel does not list {arch} explicitly. "
                "spaDIVA will test a real CUDA kernel because PTX JIT may "
                "still make the GPU usable; the first call can take several "
                "minutes.",
                RuntimeWarning,
                stacklevel=2,
            )

        probe = torch.ones(1, device="cuda")
        probe.mul_(2.0)
        torch.cuda.synchronize()
        return float(probe.cpu().item()) == 2.0
    except Exception as exc:
        warnings.warn(
            f"CUDA was detected but a test kernel failed; falling back to "
            f"CPU. Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def _resolve_device(use_cuda=False):
    return torch.device("cuda" if use_cuda and _cuda_is_usable() else "cpu")


def _as_tensor(x, device=None, dtype=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        tensor = x
    else:
        tensor = torch.tensor(x)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def train_epoch(
    model,
    X1_input,
    X2_input,
    X1_train,
    X2_train,
    edge_index=None,
    optimizer=None,
    weight=1,
    use_cuda=False,
    edge_weight=None,
    edge_index1=None,
    edge_weight1=None,
    edge_index2=None,
    edge_weight2=None,
):
    model.train()
    if edge_index1 is None:
        edge_index1 = edge_index
    if edge_weight1 is None:
        edge_weight1 = edge_weight
    if edge_index2 is None:
        edge_index2 = edge_index1
    if edge_weight2 is None:
        edge_weight2 = edge_weight1
    device = _resolve_device(use_cuda)
    X1_input = _as_tensor(X1_input, device=device, dtype=torch.float32)
    X2_input = _as_tensor(X2_input, device=device, dtype=torch.float32)
    X1_train = _as_tensor(X1_train, device=device, dtype=torch.float32)
    X2_train = _as_tensor(X2_train, device=device, dtype=torch.float32)
    edge_index1 = _as_tensor(edge_index1, device=device, dtype=torch.long)
    edge_index2 = _as_tensor(edge_index2, device=device, dtype=torch.long)
    edge_weight1 = _as_tensor(edge_weight1, device=device, dtype=torch.float32)
    edge_weight2 = _as_tensor(edge_weight2, device=device, dtype=torch.float32)
    x1_loc, x2_loc, z1_loc, z1_scale, z2_loc, z2_scale, w1_loc, w1_scale, w2_loc, w2_scale, z = model(
        X1_input, X2_input, edge_index1, edge_weight=edge_weight1, edge_index2=edge_index2, edge_weight2=edge_weight2
    )

    loss = model.loss_function(x1_loc, x2_loc, X1_train, X2_train, z1_loc, z1_scale, z2_loc, z2_scale, w1_loc, w1_scale, w2_loc, w2_scale, weight)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


def train_spadiva(
    X1_input, X2_input, X1_train, X2_train, edge_index=None,
    learning_rate=1e-3,
    weight=1.0,
    max_epochs=800,
    hidden_dim1=64,
    hidden_dim2=64,
    w1_dim=30,
    w2_dim=30,
    z_dim=30,
    KL_weight=None,
    use_cuda=False,
    edge_weight=None,
    edge_index1=None,
    edge_weight1=None,
    edge_index2=None,
    edge_weight2=None,
):

    device = _resolve_device(use_cuda)
    use_cuda = device.type == "cuda"

    VAE = VariationalAutoEncoder(
        input_dim1=X1_input.shape[1],
        input_dim2=X2_input.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        w1_dim=w1_dim,
        w2_dim=w2_dim,
        z_dim=z_dim,
        KL_weight=KL_weight,
        use_cuda=False,
    ).to(device)

    optimizer = torch.optim.AdamW(VAE.parameters(), lr=learning_rate)

    train_loss = []
    for epoch in range(max_epochs):
        total_epoch_loss_train = train_epoch(
            VAE,
            X1_input,
            X2_input,
            X1_train,
            X2_train,
            edge_index,
            optimizer,
            weight=weight,
            use_cuda=use_cuda,
            edge_weight=edge_weight,
            edge_index1=edge_index1,
            edge_weight1=edge_weight1,
            edge_index2=edge_index2,
            edge_weight2=edge_weight2,
        )
        train_loss.append(total_epoch_loss_train.cpu().detach().numpy())

    return VAE, train_loss


def infer_latents(
    model,
    X1_input,
    X2_input,
    edge_index=None,
    use_cuda=False,
    edge_weight=None,
    edge_index1=None,
    edge_weight1=None,
    edge_index2=None,
    edge_weight2=None,
    sample_seed=None,
):
    """Return latent representations and reconstructions from a trained model.

    ``sample_seed`` independently controls the sampled PoE latent returned as
    the first output without advancing the caller's PyTorch RNG state.
    """
    device = _resolve_device(use_cuda)
    model = model.to(device)
    model.eval()

    if sample_seed is None:
        rng_context = nullcontext()
    else:
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        rng_context = torch.random.fork_rng(devices=cuda_devices)

    with rng_context:
        if sample_seed is not None:
            torch.manual_seed(int(sample_seed))
        with torch.no_grad():
            if edge_index1 is None:
                edge_index1 = edge_index
            if edge_weight1 is None:
                edge_weight1 = edge_weight
            if edge_index2 is None:
                edge_index2 = edge_index1
            if edge_weight2 is None:
                edge_weight2 = edge_weight1

            x1 = _as_tensor(X1_input, device=device, dtype=torch.float32)
            x2 = _as_tensor(X2_input, device=device, dtype=torch.float32)
            eidx1 = _as_tensor(edge_index1, device=device, dtype=torch.long)
            eidx2 = _as_tensor(edge_index2, device=device, dtype=torch.long)
            ew1 = _as_tensor(edge_weight1, device=device, dtype=torch.float32)
            ew2 = _as_tensor(edge_weight2, device=device, dtype=torch.float32)

            (x1_loc, x2_loc,
             z1_loc, z1_scale,
             z2_loc, z2_scale,
             w1_loc, w1_scale,
             w2_loc, w2_scale,
             z) = model(x1, x2, eidx1, edge_weight=ew1, edge_index2=eidx2, edge_weight2=ew2)

    Z_loc = z.detach().cpu().numpy()
    Z1_loc = z1_loc.detach().cpu().numpy()
    Z2_loc = z2_loc.detach().cpu().numpy()
    W1_loc = w1_loc.detach().cpu().numpy()
    W2_loc = w2_loc.detach().cpu().numpy()
    X1_hat = x1_loc.detach().cpu().numpy()
    X2_hat = x2_loc.detach().cpu().numpy()

    return Z_loc, Z1_loc, Z2_loc, W1_loc, W2_loc, X1_hat, X2_hat


def joint_train_epoch(
    model,
    X1_ATAC, X1_RNA,
    X2_ATAC, X2_RNA,
    X3_ATAC, X3_RNA,
    edge_index1, edge_index2, edge_index3,
    match12, match23,
    joint_optimizer,
    weight=1.0,
    lambda_mnn=0.5,
    mnn_include=False,
    use_cuda=False,
):
    """Run one joint training epoch for three slices."""
    from spaDIVA.utils import compute_MNN_loss, reverse_matches

    device = _resolve_device(use_cuda)
    X1_ATAC = X1_ATAC.to(device)
    X1_RNA = X1_RNA.to(device)
    X2_ATAC = X2_ATAC.to(device)
    X2_RNA = X2_RNA.to(device)
    X3_ATAC = X3_ATAC.to(device)
    X3_RNA = X3_RNA.to(device)
    edge_index1 = edge_index1.to(device)
    edge_index2 = edge_index2.to(device)
    edge_index3 = edge_index3.to(device)

    model.train()
    x1_atac, x1_rna, z1_atac_loc, z1_atac_scale, z1_rna_loc, z1_rna_scale, w1_atac_loc, w1_atac_scale, w1_rna_loc, w1_rna_scale, z1_PoE = model(X1_ATAC, X1_RNA, edge_index1)
    x2_atac, x2_rna, z2_atac_loc, z2_atac_scale, z2_rna_loc, z2_rna_scale, w2_atac_loc, w2_atac_scale, w2_rna_loc, w2_rna_scale, z2_PoE = model(X2_ATAC, X2_RNA, edge_index2)
    x3_atac, x3_rna, z3_atac_loc, z3_atac_scale, z3_rna_loc, z3_rna_scale, w3_atac_loc, w3_atac_scale, w3_rna_loc, w3_rna_scale, z3_PoE = model(X3_ATAC, X3_RNA, edge_index3)
    
    loss1 = model.loss_function(x1_atac, x1_rna, X1_ATAC, X1_RNA, z1_atac_loc, z1_atac_scale, z1_rna_loc, z1_rna_scale, w1_atac_loc, w1_atac_scale, w1_rna_loc, w1_rna_scale, weight)
    loss2 = model.loss_function(x2_atac, x2_rna, X2_ATAC, X2_RNA, z2_atac_loc, z2_atac_scale, z2_rna_loc, z2_rna_scale, w2_atac_loc, w2_atac_scale, w2_rna_loc, w2_rna_scale, weight)
    loss3 = model.loss_function(x3_atac, x3_rna, X3_ATAC, X3_RNA, z3_atac_loc, z3_atac_scale, z3_rna_loc, z3_rna_scale, w3_atac_loc, w3_atac_scale, w3_rna_loc, w3_rna_scale, weight)

    mnn12_loss = compute_MNN_loss(z1_PoE, z2_PoE, match12) + compute_MNN_loss(z2_PoE, z1_PoE, reverse_matches(match12))
    mnn23_loss = compute_MNN_loss(z2_PoE, z3_PoE, match23) + compute_MNN_loss(z3_PoE, z2_PoE, reverse_matches(match23))

    if mnn_include is True:
        loss = loss1 + loss2 + loss3 + lambda_mnn * (mnn12_loss + mnn23_loss)
    else:
        loss = loss1 + loss2 + loss3
    loss.backward()
    joint_optimizer.step()
    joint_optimizer.zero_grad()

    return loss


def joint_train_spadiva(
    X1_ATAC,
    X1_RNA,
    X2_ATAC,
    X2_RNA,
    X3_ATAC,
    X3_RNA,
    edge_index1,
    edge_index2,
    edge_index3,
    learning_rate=1e-3,
    weight=1.0,
    max_epochs=800,
    epochs_per_update=50,
    pre_epoch=0,
    lambda_mnn=0.5,
    hidden_dim1=64,
    hidden_dim2=64,
    w1_dim=30,
    w2_dim=30,
    z_dim=30,
    use_cuda=False,
):
    
    device = _resolve_device(use_cuda)
    use_cuda = device.type == "cuda"

    VAE = VariationalAutoEncoder(
        input_dim1=X1_ATAC.shape[1],
        input_dim2=X1_RNA.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        w1_dim=w1_dim,
        w2_dim=w2_dim,
        z_dim=z_dim,
        use_cuda=False,
    ).to(device)

    joint_optimizer = torch.optim.AdamW(VAE.parameters(),lr=learning_rate)

    train_loss = []
    for epoch in range(max_epochs):
        if epoch % epochs_per_update == 0:
            from spaDIVA.utils import update_mnn

            z1_PoE, z1_atac_loc, z1_rna_loc, w1_atac_loc, w1_rna_loc, x1_atac, x1_rna = infer_latents(VAE, X1_ATAC, X1_RNA, edge_index1, use_cuda=use_cuda)
            z2_PoE, z2_atac_loc, z2_rna_loc, w2_atac_loc, w2_rna_loc, x2_atac, x2_rna = infer_latents(VAE, X2_ATAC, X2_RNA, edge_index2, use_cuda=use_cuda)
            z3_PoE, z3_atac_loc, z3_rna_loc, w3_atac_loc, w3_rna_loc, x3_atac, x3_rna = infer_latents(VAE, X3_ATAC, X3_RNA, edge_index3, use_cuda=use_cuda)

            a1, b1 = cal_weight(z1_atac_loc, z1_rna_loc, k=20)
            Z1_WNN = a1.reshape(-1, 1) * z1_atac_loc + b1.reshape(-1, 1) * z1_rna_loc
            a2, b2 = cal_weight(z2_atac_loc, z2_rna_loc, k=20)
            Z2_WNN = a2.reshape(-1, 1) * z2_atac_loc + b2.reshape(-1, 1) * z2_rna_loc
            a3, b3 = cal_weight(z3_atac_loc, z3_rna_loc, k=20)
            Z3_WNN = a3.reshape(-1, 1) * z3_atac_loc + b3.reshape(-1, 1) * z3_rna_loc
            match12 = update_mnn(Z1_WNN, Z2_WNN)
            match23 = update_mnn(Z2_WNN, Z3_WNN)

        if epoch >= pre_epoch:
            mnn_include = True
        else:
            mnn_include = False

        total_epoch_loss_train = joint_train_epoch(
            VAE,
            _as_tensor(X1_ATAC, dtype=torch.float32), _as_tensor(X1_RNA, dtype=torch.float32),
            _as_tensor(X2_ATAC, dtype=torch.float32), _as_tensor(X2_RNA, dtype=torch.float32),
            _as_tensor(X3_ATAC, dtype=torch.float32), _as_tensor(X3_RNA, dtype=torch.float32),
            _as_tensor(edge_index1, dtype=torch.long),
            _as_tensor(edge_index2, dtype=torch.long),
            _as_tensor(edge_index3, dtype=torch.long),
            match12, match23,
            joint_optimizer,
            weight=weight,
            lambda_mnn=lambda_mnn,
            mnn_include=mnn_include,
            use_cuda=use_cuda,
        )
        train_loss.append(total_epoch_loss_train.cpu().detach().numpy())

    return VAE, train_loss
