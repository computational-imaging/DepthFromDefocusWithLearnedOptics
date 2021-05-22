import torch
import torch.testing

from util import cubicspline

torch.manual_seed(0)


def test_interp():
    n = 10
    c = 3
    d = 4
    t = 5
    x = torch.linspace(0, 1, steps=n)
    x = x[None, :].repeat((c, 1))
    y = torch.randn(c, d, n)
    xs = x[:, t].reshape(c, 1, 1)
    ind = torch.tensor(t, dtype=torch.long).reshape(1, 1, 1).repeat((c, 1, 1))
    ys = cubicspline.interp(x, y, xs, ind)
    yy = torch.stack([y[i, :, ind[i]] for i in range(c)], dim=0)
    torch.testing.assert_allclose(yy, ys)
