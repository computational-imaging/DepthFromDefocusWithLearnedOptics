import torch
import torch.testing

from util.helper import matting


def test_matting():
    # binary (ceiling the index)
    torch.testing.assert_allclose(
        matting(torch.tensor(0.), 4, binary=True).squeeze(),
        torch.tensor([1., 0., 0., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1 / 4), 4, binary=True).squeeze(),
        torch.tensor([1., 0., 0., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(2 / 4), 4, binary=True).squeeze(),
        torch.tensor([0., 1., 0., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(3 / 4), 4, binary=True).squeeze(),
        torch.tensor([0., 0., 1., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1.), 4, binary=True).squeeze(),
        torch.tensor([0., 0., 0., 1.])
    )

    # non-binary
    torch.testing.assert_allclose(
        matting(torch.tensor(0.), 4, binary=False).squeeze(),
        torch.tensor([1., 0., 0., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1 / 4), 4, binary=False).squeeze(),
        torch.tensor([1., 1., 0., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(2 / 4), 4, binary=False).squeeze(),
        torch.tensor([0., 1., 1., 0.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(3 / 4), 4, binary=False).squeeze(),
        torch.tensor([0., 0., 1., 1.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1.), 4, binary=False).squeeze(),
        torch.tensor([0., 0., 0., 1.])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1. - 1 / 4 * 0.1), 4, binary=False).squeeze(),
        torch.tensor([0., 0., 0.1, 1.0])
    )
    torch.testing.assert_allclose(
        matting(torch.tensor(1. - 1 / 4 * 0.9), 4, binary=False).squeeze(),
        torch.tensor([0., 0., 0.9, 1.0])
    )
