import itertools

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal

from solvers import image_reconstruction
from util import complex

torch.manual_seed(0)


def test_compute_regularized_normal_inv():
    K = 5
    beta = 1.5
    compvec = torch.rand((K, 2))
    compeye = complex.eye(K)
    M = complex.mul_with_func(compvec, complex.conj(compvec), torch.ger) + beta * compeye
    invM = image_reconstruction.compute_regularized_normal_inv(compvec, beta)
    MinvM = complex.mul_with_func(M, invM, torch.mm).numpy()

    c = compvec.numpy()
    c = c[:, 0] + 1j * c[:, 1]
    M_np = np.ma.outer(c, np.conj(c)) + beta * np.eye(K)
    invM_np = np.linalg.inv(M_np)

    assert_almost_equal(M[..., 0], np.real(M_np), decimal=6)
    assert_almost_equal(M[..., 1], np.imag(M_np), decimal=6)
    assert_almost_equal(invM[..., 0], np.real(invM_np), decimal=6)
    assert_almost_equal(invM[..., 1], np.imag(invM_np), decimal=6)
    assert_almost_equal(MinvM[..., 0], np.eye(K), decimal=6)
    assert_almost_equal(MinvM[..., 1], np.zeros((K, K)), decimal=6)


testparams_tikhonov = (False, True)


def test_tikhonov_inverse():
    eps = 1e-2
    H, W, D = 10, 10, 6
    gt_x = torch.zeros((D, H, W))
    gt_x[0, 1:3, 1:3] = 1

    psf = torch.rand((D, H, W))
    psf /= psf.sum(dim=(1, 2), keepdim=True)
    beta = 1.

    def image_formation(x, g):
        X = torch.rfft(x, 2)
        G = torch.rfft(g, 2)
        XG = complex.multiply(X, G)
        y = torch.sum(torch.irfft(XG, 2, signal_sizes=(H, W)), dim=0)
        return y

    y = image_formation(gt_x, psf)

    est_x = image_reconstruction.tikhonov_inverse(y, psf, beta, eps)

    def loss(input):
        Ax = image_formation(input, psf)
        reg = torch.sum(input ** 2)
        return torch.sum((Ax - y) ** 2) + beta * reg + eps * torch.sum(input ** 2)

    param = torch.nn.Parameter(est_x)
    loss_val = loss(param)
    loss_val.backward()
    assert param.grad.abs().max().item() < 1e-5


testparams_tikhonov_fast = itertools.product((1, 2), (False, True))


@pytest.mark.parametrize('num_shots,tikhonov_reg', testparams_tikhonov_fast)
def test_tikhonov_inverse_fast(num_shots, tikhonov_reg):
    beta = 2.0
    gamma = 5.0

    K, D, H, W = num_shots, 6, 100, 100
    gt_x = torch.zeros((D, H, W))
    gt_x[0, 1:3, 1:3] = 0.5
    gt_x[1, 6:9, 4:6] = 0.2

    psf = torch.rand((K, D, H, W))
    psf /= psf.sum(dim=(2, 3), keepdim=True)

    def image_formation(x, g):
        x = x.unsqueeze(0)
        X = torch.rfft(x, 2)
        G = torch.rfft(g, 2)
        XG = complex.multiply(X, G)
        y = torch.sum(torch.irfft(XG, 2, signal_sizes=(H, W)), dim=1)
        return y

    if tikhonov_reg:
        v = gt_x + 0.001 * torch.randn((D, H, W))
    else:
        v = None

    y = image_formation(gt_x, psf)
    Y = torch.rfft(y, 2)
    G = torch.rfft(psf, 2)
    est_X = image_reconstruction.tikhonov_inverse_fast(Y, G, v, beta, gamma)
    est_x = torch.irfft(est_X, 2, signal_sizes=(H, W))

    def loss(input):
        Ax = image_formation(input, psf)
        if tikhonov_reg:
            reg = gamma * torch.sum((input - v) ** 2)
        else:
            reg = gamma * torch.sum(input ** 2)
        return torch.sum((Ax - y) ** 2) + reg

    param = torch.nn.Parameter(est_x)
    loss_val = loss(param)
    loss_val.backward()
    assert param.grad.abs().max().item() < 1e-5
