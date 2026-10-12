import dataclasses

import numpy as np
import pytest
import torch

from pixelwm.envs import PointReach
from pixelwm.eval import open_loop_eval
from pixelwm.models import VQTransformer, VQTransformerConfig, WorldModel, build_model
from pixelwm.planning import CEMPlanner, evaluate_policy
from pixelwm.utils.checkpoint import load_checkpoint, save_checkpoint

SMALL = VQTransformerConfig(
    num_codes=16, code_dim=8, cnn_depth=8, window=2, d_model=32, n_layers=1, n_heads=2,
    tokenizer_steps=2, tokenizer_batch=16, window_batch=8,
)


@pytest.fixture
def model() -> VQTransformer:
    torch.manual_seed(0)
    return VQTransformer(action_dim=2, config=SMALL)


@pytest.fixture
def batch(small_buffer, rng):
    return small_buffer.sample(batch_size=3, seq_len=8, rng=rng)


def test_is_world_model(model):
    assert isinstance(model, WorldModel) and model.supports_decoding


def test_observe_shapes_and_first_frame(model, batch):
    post, prior = model.observe(batch["obs"], batch["prev_actions"])
    assert post["tokens"].shape == (3, 8, 2, 64) and post["tokens"].dtype == torch.long
    assert post["actions"].shape == (3, 8, 2, 2) and post["reward"].shape == (3, 8)
    assert prior["tokens"].shape == post["tokens"].shape
    # at the start of an episode the window is filled with copies of the first frame
    assert torch.equal(post["tokens"][:, 0, 0], post["tokens"][:, 0, 1])
    assert torch.all(post["actions"][:, 0] == 0) and torch.all(post["reward"][:, 0] == 0)
    # the posterior holds the real tokens of each frame
    assert torch.equal(post["tokens"][:, :, -1], model.tokenize(batch["obs"]))


def test_observe_can_continue_from_a_state(model, batch):
    full, _ = model.observe(batch["obs"], batch["prev_actions"])
    head, _ = model.observe(batch["obs"][:, :4], batch["prev_actions"][:, :4])
    state = {k: v[:, -1] for k, v in head.items()}
    tail, _ = model.observe(batch["obs"][:, 4:], batch["prev_actions"][:, 4:], state)
    assert torch.equal(full["tokens"][:, 4:], tail["tokens"])


def test_imagine_decode_and_reward_shapes(model):
    state = model.initial_state(2)
    imagined = model.imagine(state, torch.zeros(2, 5, 2))
    assert imagined["tokens"].shape == (2, 5, 2, 64) and imagined["reward"].shape == (2, 5)
    images = model.decode(imagined)
    assert images.shape == (2, 5, 3, 64, 64) and images.min() >= 0.0 and images.max() <= 1.0
    assert model.predict_reward(imagined).shape == (2, 5)
    assert model.features(model.initial_state(2)).shape == (2, 64 * SMALL.code_dim)


def test_greedy_decoding_is_deterministic(model):
    model.eval()
    model.temperature = 0.0
    state, actions = model.initial_state(2), torch.randn(2, 3, 2)
    a, b = model.imagine(state, actions), model.imagine(state, actions)
    assert torch.equal(a["tokens"], b["tokens"])


def test_two_stage_training(model, batch):
    assert model.stage == "tokenizer"
    for expected in ("recon_loss", "recon_loss", "token_loss"):
        model.zero_grad(set_to_none=True)
        loss, metrics = model.loss(batch)
        assert expected in metrics and torch.isfinite(loss)
        loss.backward()
    assert int(model.train_calls) == 3 and model.stage == "transformer"
    # the last call trained the transformer on frozen tokens
    assert all(p.grad is None for p in model.tokenizer.parameters())
    assert any(p.grad is not None for p in model.token_head.parameters())
    assert {"token_acc", "frame_acc", "reward_loss"} <= metrics.keys()


def test_episode_start_augmentation_zeroes_the_first_action():
    cfg = dataclasses.replace(SMALL, pad_prob=1.0, window_batch=32)
    model = VQTransformer(2, cfg)
    tokens = torch.randint(0, 16, (3, 8, 64))
    actions = torch.randn(3, 8, 2) + 5.0  # never exactly zero
    windows = model._make_windows(tokens, actions, torch.zeros(3, 8))
    assert windows["tokens"].shape == (32, 2, 64) and windows["actions"].shape == (32, 2, 2)
    assert windows["target"].shape == (32, 64) and windows["action"].shape == (32, 2)
    assert torch.all(windows["actions"][:, 0] == 0)  # the oldest frame is always "episode start"


def test_window_longer_than_sequence_raises():
    model = VQTransformer(2, dataclasses.replace(SMALL, window=4))
    with pytest.raises(ValueError):
        tokens = torch.zeros(2, 4, 64, dtype=torch.long)
        model._make_windows(tokens, torch.zeros(2, 4, 2), torch.zeros(2, 4))


def test_transformer_stage_learns_on_a_fixed_batch(batch):
    torch.manual_seed(0)
    cfg = dataclasses.replace(SMALL, tokenizer_steps=0, window_batch=32)
    model = VQTransformer(2, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, metrics = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(metrics["token_loss"])
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_checkpoint_roundtrip(model, tmp_path):
    save_checkpoint(tmp_path / "m.pt", model, step=3, extra={"env": {"name": "point_reach"}})
    restored, ckpt = load_checkpoint(tmp_path / "m.pt")
    assert ckpt["model_name"] == "vq_transformer" and restored.config == model.config
    original, loaded = model.state_dict(), restored.state_dict()
    assert original.keys() == loaded.keys()
    for key, value in original.items():
        assert torch.equal(value, loaded[key]), key


def test_registry_builds_the_model():
    built = build_model("vq_transformer", 2, dataclasses.asdict(SMALL))
    assert isinstance(built, VQTransformer)


def test_open_loop_eval_with_object_and_reward_metrics(model, small_buffer):
    results = open_loop_eval(
        model, small_buffer, horizons=(1, 4), context=3, num_sequences=4, object_metrics=True
    )
    for key in ("psnr@1", "ssim@4", "reward_mae@4", "agent_found@1", "goal_err_copy@4"):
        assert key in results
    assert np.isfinite(results["psnr@1"]) and np.isfinite(results["reward_mae@1"])


def test_cem_planner_runs_with_the_token_model(model):
    planner = CEMPlanner(model, horizon=2, num_samples=8, num_elites=2, iterations=1)
    env = PointReach(max_steps=3)
    result = evaluate_policy(env, planner, episodes=1, seed=0)
    assert len(result["returns"]) == 1 and np.isfinite(result["return_mean"])
