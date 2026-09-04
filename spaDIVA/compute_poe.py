import torch

__all__ = ["compute_poe_gaussian"]


def compute_poe_gaussian(z1_loc, z1_scale, z2_loc, z2_scale):
    """Combine two diagonal Gaussian distributions by product of experts."""
    z1_var = torch.exp(z1_scale)
    z2_var = torch.exp(z2_scale)

    poe_var = 1 / (1 / z1_var + 1 / z2_var)
    poe_loc = poe_var * (z1_loc / z1_var + z2_loc / z2_var)
    poe_scale = torch.log(poe_var)

    return poe_loc, poe_scale
