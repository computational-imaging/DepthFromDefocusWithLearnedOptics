import math

import torch
import torch.nn.functional as F


def matting(depthmap, n_depths, binary, eps=1e-8):
    depthmap = depthmap.clamp(eps, 1.0)
    d = torch.arange(0, n_depths, dtype=depthmap.dtype, device=depthmap.device).reshape(1, 1, -1, 1, 1) + 1
    depthmap = depthmap * n_depths
    diff = d - depthmap
    alpha = torch.zeros_like(diff)
    if binary:
        alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        alpha[mask] = diff[mask] + 1.
        alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.
    return alpha


def depthmap_to_layereddepth(depthmap, n_depths, binary=False):
    depthmap = depthmap[:, None, ...]  # add color dim
    layered_depth = matting(depthmap, n_depths, binary=binary)
    return layered_depth


def over_op(alpha):
    bs, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, cs, 1, hs, ws), dtype=out.dtype, device=out.device), out[:, :, :-1]], dim=-3)


def crop_boundary(x, w):
    if w == 0:
        return x
    else:
        return x[..., w:-w, w:-w]


def refractive_index(wavelength, a=1.5375, b=0.00829045, c=-0.000211046):
    """Cauchy's equation - dispersion formula
    Default coefficients are for NOA61.
    https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def linear_to_srgb(x, eps=1e-8):
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)


def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def heightmap_to_phase(height, wavelength, refractive_index):
    return height * (2 * math.pi / wavelength) * (refractive_index - 1)


def phase_to_heightmap(phase, wavelength, refractive_index):
    return phase / (2 * math.pi / wavelength) / (refractive_index - 1)


def imresize(img, size):
    return F.interpolate(img, size=size)


def ips_to_metric(d, min_depth, max_depth):
    """
    https://github.com/fyu/tiny/blob/4572a056fd92696a3a970c2cffd3ba1dae0b8ea0/src/sweep_planes.cc#L204

    Args:
        d: inverse perspective sampling [0, 1]
        min_depth: in meter
        max_depth: in meter

    Returns:

    """
    return (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * d)


def metric_to_ips(d, min_depth, max_depth):
    """

    Args:
        d: metric depth [min_depth, max_depth]
        min_dpeth: in meter
        max_depth: in meter

    Returns:
    """
    # d = d.clamp(min_depth, max_depth)
    return (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)


def copy_quadruple(x_rd):
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    x = torch.cat([x_u, x_d], dim=-1)
    return x


def to_bayer(x):
    mask = torch.zeros_like(x)
    # masking r
    mask[:, 0, ::2, ::2] = 1
    # masking b
    mask[:, 2, 1::2, 1::2] = 1
    # masking g
    mask[:, 1, 1::2, ::2] = 1
    mask[:, 1, ::2, 1::2] = 1
    y = x * mask
    bayer = y.sum(dim=1, keepdim=True)
    return bayer
