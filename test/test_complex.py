import numpy as np
import torch

from util import complex


def test_multiply():
    n = 10
    x_np = np.random.rand(n) + 1j * np.random.rand(n)
    y_np = np.random.rand(n) + 1j * np.random.rand(n)
    xy_np = x_np * y_np
    xystack_np = np.stack([np.real(xy_np), np.imag(xy_np)], axis=-1)
    x = torch.stack([torch.from_numpy(np.real(x_np)), torch.from_numpy(np.imag(x_np))], dim=-1)
    y = torch.stack([torch.from_numpy(np.real(y_np)), torch.from_numpy(np.imag(y_np))], dim=-1)
    xystack = complex.multiply(x, y)
    torch.testing.assert_allclose(xystack_np, xystack)


def test_conj():
    n = 10
    x_np = np.random.rand(n) + 1j * np.random.rand(n)
    xconj_np = np.conj(x_np)
    x = torch.stack([torch.from_numpy(np.real(x_np)), torch.from_numpy(np.imag(x_np))], dim=-1)
    xconjstack_np = torch.stack([torch.from_numpy(np.real(xconj_np)), torch.from_numpy(np.imag(xconj_np))], dim=-1)
    xconj = complex.conj(x)
    torch.testing.assert_allclose(xconjstack_np, xconj)


def test_abs2():
    n = 10
    x_np = np.random.rand(n) + 1j * np.random.rand(n)
    xabs2_np = np.abs(x_np) ** 2
    xabs2_np = torch.from_numpy(xabs2_np)
    x = torch.stack([torch.from_numpy(np.real(x_np)), torch.from_numpy(np.imag(x_np))], dim=-1)
    xabs2 = complex.abs2(x)
    torch.testing.assert_allclose(xabs2_np, xabs2)
