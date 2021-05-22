import itertools

import numpy as np
import pytest
import torch

from util.fft import fftshift, ifftshift

size = ((3,), (4,), (4, 4), (3, 4), (4, 3), (4, 5, 6))


@pytest.mark.parametrize('size', size)
def test_fftshift(size):
    ndims = len(size)
    x = torch.rand(size)
    x_np = x.numpy()
    for d in range(ndims):
        for axes in itertools.combinations(range(ndims), d + 1):
            y = fftshift(x, axes)
            y_np = np.fft.fftshift(x_np, axes)
            print(axes, size)
            print(x, '\n', x_np)
            print(y, '\n', y_np)
            torch.testing.assert_allclose(y, y_np)


@pytest.mark.parametrize('size', size)
def test_ifftshift(size):
    ndims = len(size)
    x = torch.rand(size)
    x_np = x.numpy()
    for d in range(ndims):
        for axes in itertools.combinations(range(ndims), d + 1):
            y = ifftshift(x, axes)
            y_np = np.fft.ifftshift(x_np, axes)
            torch.testing.assert_allclose(y, y_np)
