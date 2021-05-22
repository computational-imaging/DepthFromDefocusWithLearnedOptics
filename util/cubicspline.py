import torch


def poly_coeff(t):
    t1 = t
    t2 = t ** 2
    t3 = t ** 3
    return (2 * t3 - 3 * t2 + 1), (t3 - 2 * t2 + t1), (-2 * t3 + 3 * t2), (t3 - t2)


def interp(x, y, xs, ind):
    """
    ind: left index of xs in x
    Assumes uniform sampling. It could be arbitrary sampling, but not implemented.
    """
    n, h, w = ind.shape

    dx = x[:, [1]] - x[:, [0]]
    diff = torch.stack([xs[i] - x[i, ind[i]] for i in torch.arange(n)], dim=0)

    c0, c1, c2, c3 = poly_coeff(diff.unsqueeze(1) / dx.reshape(-1, 1, 1, 1))

    # Add depth dimension
    x = x.unsqueeze(1)
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1]) / dx.reshape(-1, 1, 1)
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1)

    y0 = torch.stack([y[i, :, ind[i]] for i in torch.arange(n)], dim=0)
    y1 = torch.stack([y[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)
    m0 = torch.stack([m[i, :, ind[i]] for i in torch.arange(n)], dim=0)
    m1 = torch.stack([m[i, :, ind[i] + 1] for i in torch.arange(n)], dim=0)

    ys = c0 * y0 + c1 * m0 + c2 * y1 + c3 * m1

    return ys
