import torch
import torch.nn.functional as F

from util import complex
from util.edgetaper import edgetaper3d



def compute_regularized_normal_inv(Hk, beta):
    K = Hk.shape[0]
    outer_prod = complex.mul_with_func(Hk, complex.conj(Hk), torch.ger)
    inner_prod = complex.mul_with_func(Hk, complex.conj(Hk), torch.dot)[0]  # imaginary part should be zero
    return 1 / beta * (complex.eye(K) - outer_prod / (beta + inner_prod))


def tikhonov_inverse(y, g, beta, eps=1e-2):
    """
    Compute Laplacian-regularized Tiknov inverse. This function solves
        argmin_x || y - sum_{k} G_k x_k ||^2 + beta sum_{k} || S x_k ||^2 + eps || x ||^2
    S: 2D Laplacian operator
    x_k: k-th 2D slice of a 3D volume x

    eps || x ||^2 seems to be required for numerical stability.
    """
    Y = torch.rfft(y, 2)[None, :, :, :]
    G = torch.rfft(g, 2)
    Gc = complex.conj(G)
    GcY = complex.multiply(Gc, Y)
    X = torch.zeros_like(G)
    # I could probably use torch.bmm to avoid creating X but am leaving it for now.
    for h in range(G.shape[1]):
        for w in range(G.shape[2]):
            reg = beta + eps
            Gk = Gc[:, h, w, :]
            invM = compute_regularized_normal_inv(Gk, reg)
            X[:, h, w, :] = complex.mul_with_func(invM, GcY[:, h, w, :], torch.mv)
    return torch.irfft(X, 2, signal_sizes=g.shape[1:3])


def tikhonov_inverse_fast(Y, G, v=None, beta=0, gamma=1e-1, dataformats='SDHW'):
    """
    Compute Tikhonov-regularized inverse. This function solves
        argmin_x || y - sum_{k} G_k x_k ||^2 + beta sum_{k} || S x_k ||^2 + gamma || x - v ||^2
    x_k: k-th 2D slice of a 3D volume x
    S: 2D Laplacian operator

    gamma || x ||^2 seems to be required for numerical stability.

    original signal size is assumed to be even.

    Even though the element in the block-diagonal matrix in Woodbuery formula is described as a row vector in our
    equation, the implementation uses a vector for convenience. Be careful about its conjugate transpose.
    """
    if dataformats == 'DHW':
        Y = Y[None, None, None, ...]  # add batch, color and shot dimension
        G = G[None, None, None, ...]  # add batch, color and shot dimension
        if v is not None:
            v = v[None, None, None, ...]
    elif dataformats == 'SDHW':
        Y = Y[None, None, ...]  # add batch and color dimension
        G = G[None, None, ...]  # add batch and color dimension
        if v is not None:
            v = v[None, None, ...]
    elif dataformats == 'CSDHW':
        Y = Y[None, ...]  # add batch dimension
        G = G[None, ...]  # add batch dimension
        if v is not None:
            v = v[None, ...]
    elif dataformats == 'BCDHW':
        Y = Y.unsqueeze(2)  # add shot dimension
        G = G.unsqueeze(2)
        if v is not None:
            v = v.unsqueeze(2)
    elif dataformats == 'BCSDHW':
        pass
    else:
        raise NotImplementedError(f'This data format is not supported! [dataformats: {dataformats}]')

    device = Y.device
    dtype = Y.dtype
    num_colors, num_shots, depth, height, width = G.shape[1:6]
    batch_sz = Y.shape[0]

    Y_real = Y[..., 0].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    Y_imag = Y[..., 1].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    G_real = (G[..., 0]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    G_imag = (G[..., 1]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    Gc_real = G_real
    Gc_imag = -G_imag

    GcY_real = (Gc_real * Y_real - Gc_imag * Y_imag).sum(dim=-1, keepdims=True)
    GcY_imag = (Gc_imag * Y_real + Gc_real * Y_imag).sum(dim=-1, keepdims=True)

    # This part is still not covered in test!
    if v is not None:
        V = gamma * torch.rfft(v, 2)
        V_real = (V[..., 0]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        V_imag = (V[..., 1]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        GcY_real += V_real
        GcY_imag += V_imag

    if not isinstance(gamma, torch.Tensor):
        reg = torch.tensor(gamma, device=device, dtype=dtype)
    else:
        reg = gamma

    Gc_real_t = Gc_real.transpose(3, 4)
    Gc_imag_t = Gc_imag.transpose(3, 4)
    # innerprod's imaginary part should be zero.
    # The conjugate transpose is implicitly reflected in the sign of complex multiplication.
    if num_shots == 1:
        innerprod = torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag)
        outerprod_real = torch.matmul(G_real, Gc_real_t) - torch.matmul(G_imag, Gc_imag_t)
        outerprod_imag = torch.matmul(G_imag, Gc_real_t) + torch.matmul(G_real, Gc_imag_t)
        invM_real = 1. / reg * (
                torch.eye(depth, device=device, dtype=dtype) - outerprod_real / (reg + innerprod))
        invM_imag = -1. / reg * outerprod_imag / (reg + innerprod)
    else:
        eye_plus_inner = torch.eye(num_shots, device=device, dtype=dtype) + 1 / reg * (
                torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag))
        eye_plus_inner_inv = torch.inverse(eye_plus_inner)
        inner_Gc_real = torch.matmul(eye_plus_inner_inv, Gc_real_t)
        inner_Gc_imag = torch.matmul(eye_plus_inner_inv, Gc_imag_t)
        prod_real = 1 / reg * (torch.matmul(G_real, inner_Gc_real) - torch.matmul(G_imag, inner_Gc_imag))
        prod_imag = 1 / reg * (torch.matmul(G_imag, inner_Gc_real) + torch.matmul(G_real, inner_Gc_imag))
        invM_real = 1 / reg * (torch.eye(depth, device=device, dtype=dtype).unsqueeze(0) - prod_real)
        invM_imag = - 1 / reg * prod_imag

    X_real = (torch.matmul(invM_real, GcY_real) - torch.matmul(invM_imag, GcY_imag))
    X_imag = (torch.matmul(invM_imag, GcY_real) + torch.matmul(invM_real, GcY_imag))
    X = torch.stack(
        [X_real.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width),
         X_imag.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width)],
        dim=-1)

    if dataformats == 'SDHW':
        X = X.reshape(depth, height, width, 2)
    elif dataformats == 'CSDHW':
        X = X.reshape(num_colors, depth, height, width, 2)
    elif dataformats == 'BCSDHW' or dataformats == 'BCDHW':
        X = X.reshape(batch_sz, num_colors, depth, height, width, 2)

    return X


def apply_tikhonov_inverse(captimg, psf, reg_tikhonov, apply_edgetaper=True):
    """

    Args:
        captimg: (B x C x H x W)
        psf: PSF lateral size should be equal to captimg. (1 x C x D x H x W)
        reg_tikhonov: float

    Returns:
        B x C x D x H x W

    """
    if apply_edgetaper:
        # Edge tapering
        captimg = edgetaper3d(captimg, psf)
    Fpsf = torch.rfft(psf, 2)
    Fcaptimgs = torch.rfft(captimg, 2)
    Fpsf = Fpsf.unsqueeze(2)  # add shot dim
    Fcaptimgs = Fcaptimgs.unsqueeze(2)  # add shot dim
    est_X = tikhonov_inverse_fast(Fcaptimgs, Fpsf, v=None, beta=0, gamma=reg_tikhonov,
                                  dataformats='BCSDHW')
    est_volumes = torch.irfft(est_X, 2, signal_sizes=captimg.shape[-2:])
    return est_volumes
