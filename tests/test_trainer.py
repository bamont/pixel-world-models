import json

from pixelwm.models import RSSM, RSSMConfig
from pixelwm.training import TrainConfig, Trainer


def test_trainer_smoke(small_buffer, tmp_path):
    model = RSSM(2, RSSMConfig(deter=16, groups=2, classes=4, hidden=16, cnn_depth=8))
    cfg = TrainConfig(
        steps=4, batch_size=2, seq_len=8, log_every=2, eval_every=4, save_every=4,
        eval_context=3, eval_horizons=(1, 3), eval_sequences=2,
    )
    trainer = Trainer(model, small_buffer, cfg, tmp_path, val_buffer=small_buffer)
    trainer.fit()
    assert (tmp_path / "checkpoints" / "last.pt").exists()
    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any("val/psnr@1" in line for line in lines)
