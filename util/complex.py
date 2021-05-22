import torch


def pack(real, imag):
    return torch.stack([real, imag], dim=-1)


def unpack(x):
    return x[..., 0], x[..., 1]


def conj(x):
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def ones(shape, dtype=torch.float32, device=torch.device('cpu')):
    return torch.stack([torch.ones(shape, dtype=dtype, device=device),
                        torch.zeros(shape, dtype=dtype, device=device)], dim=-1)


def eye(K):
    return torch.stack([torch.eye(K), torch.zeros((K, K))], dim=-1)


def abs2(x):
    return x[..., -1] ** 2 + x[..., -2] ** 2


def multiply(x, y):
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    return torch.stack([x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag], dim=-1)


def mul_with_func(x, y, func):
    x_real, x_imag = unpack(x)
    y_real, y_imag = unpack(y)
    xr_yr = func(x_real, y_real)
    xr_yi = func(x_real, y_imag)
    xi_yr = func(x_imag, y_real)
    xi_yi = func(x_imag, y_imag)
    real = xr_yr - xi_yi
    imag = xr_yi + xi_yr
    return torch.stack([real, imag], dim=-1)
