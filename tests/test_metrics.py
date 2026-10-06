import torch

from pixelwm.eval import psnr, ssim


def test_identical_images():
    x = torch.rand(2, 3, 64, 64)
    assert torch.all(psnr(x, x) > 90)
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5)


def test_noise_lowers_scores():
    x = torch.rand(2, 3, 64, 64)
    noisy = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)
    assert torch.all(psnr(noisy, x) < psnr(x, x))
    assert torch.all(ssim(noisy, x) < 0.99)
