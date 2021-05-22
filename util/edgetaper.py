"""
Refer to
https://github.com/AndreiDavydov/Poisson_Denoiser/blob/master/pydl/nnLayers/functional/functional.py
under MIT Licence (copyright: Andrei Davydov)
"""
import torch

from util import complex


def autocorrelation1d_symmetric(h):
    """Compute autocorrelation of a symmetric signal along the last dimension"""
    Fhsq = complex.abs2(torch.rfft(h, 1))
    a = torch.irfft(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=(h.shape[-1],))
    return a / a.max()


def compute_weighting_for_tapering(h):
    """Compute autocorrelation of a symmetric signal along the last two dimension"""
    h_proj0 = h.sum(dim=-2, keepdims=False)
    autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2)
    h_proj1 = h.sum(dim=-1, keepdims=False)
    autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1)
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)


def edgetaper3d(img, psf):
    """
    Edge-taper an image with a depth-dependent PSF

    Args:
        img: (B x C x H x W)
        psf: 3d rotationally-symmetric psf (B x C x D x H x W) (i.e. continuous at boundaries)

    Returns:
        Edge-tapered 3D image
    """
    assert (img.dim() == 4)
    assert (psf.dim() == 5)
    psf = psf.mean(dim=-3)
    alpha = compute_weighting_for_tapering(psf)
    blurred_img = torch.irfft(
        complex.multiply(torch.rfft(img, 2), torch.rfft(psf, 2)), 2, signal_sizes=img.shape[-2:]
    )
    return alpha * img + (1 - alpha) * blurred_img
