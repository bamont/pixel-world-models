import numpy as np
import pytest
import torch

from pixelwm.eval import open_loop_eval
from pixelwm.models import RSSM, RSSMConfig, WorldModel, build_model
from pixelwm.models.rssm import categorical_kl
from pixelwm.utils.checkpoint import load_checkpoint, save_checkpoint

SMALL = RSSMConfig(deter=32, groups=4, classes=4, hidden=32, cnn_depth=8)


@pytest.fixture
def model() -> RSSM:
    torch.manual_seed(0)
    return RSSM(action_dim=2, config=SMALL)


@pytest.fixture
def batch(small_buffer, rng):
    return small_buffer.sample(batch_size=3, seq_len=8, rng=rng)


def test_is_world_model(model):
    assert isinstance(model, WorldModel) and model.supports_decoding


def test_observe_shapes(model, batch):
    post, prior = model.observe(batch["obs"], batch["prev_actions"])
    assert post["deter"].shape == (3, 8, 32)
    assert post["stoch"].shape == (3, 8, 16)
    assert prior["logits"].shape == (3, 8, 4, 4)


def test_imagine_and_decode_shapes(model):
    state = model.initial_state(2)
    imagined = model.imagine(state, torch.zeros(2, 6, 2))
    assert imagined["deter"].shape == (2, 6, 32)
    assert model.decode(imagined).shape == (2, 6, 3, 64, 64)
    assert model.predict_reward(imagined).shape == (2, 6)


def test_decode_range(model):
    img = model.decode(model.initial_state(2))
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_loss_is_finite_and_backpropagates(model, batch):
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert {"loss", "recon_loss", "reward_loss", "kl"} <= metrics.keys()


def test_loss_decreases_when_overfitting_one_batch(model, batch):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = model.loss(batch)[1]["recon_loss"]
    for _ in range(30):
        loss, _ = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert model.loss(batch)[1]["recon_loss"] < first


def test_categorical_kl_zero_for_identical():
    logits = torch.randn(5, 4, 4)
    assert torch.allclose(categorical_kl(logits, logits), torch.zeros(5), atol=1e-6)
    assert torch.all(categorical_kl(logits, torch.randn(5, 4, 4)) >= 0)


def test_open_loop_eval_keys(model, small_buffer):
    results = open_loop_eval(
        model, small_buffer, horizons=(1, 5), context=3, num_sequences=4, seed=0
    )
    assert all(np.isfinite(v) for v in results.values())
    expected = {f"{m}@{h}" for m in ("psnr", "ssim", "mse") for h in (1, 5)}
    assert expected <= set(results)


def test_checkpoint_roundtrip(model, tmp_path):
    save_checkpoint(tmp_path / "m.pt", model, step=7, extra={"env": {"name": "point_reach"}})
    restored, ckpt = load_checkpoint(tmp_path / "m.pt")
    assert ckpt["step"] == 7 and restored.config == model.config
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_registry_unknown_model():
    with pytest.raises(KeyError):
        build_model("nope", 2)
