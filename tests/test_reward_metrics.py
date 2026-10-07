import numpy as np

from pixelwm.eval import open_loop_eval
from pixelwm.models import RSSM, RSSMConfig


def test_reward_metrics_are_reported(small_buffer):
    model = RSSM(2, RSSMConfig(deter=16, groups=2, classes=4, hidden=16, cnn_depth=8))
    results = open_loop_eval(model, small_buffer, horizons=(1, 4), context=3, num_sequences=6)
    for h in (1, 4):
        assert np.isfinite(results[f"reward_mae@{h}"]) and results[f"reward_mae@{h}"] >= 0
        assert np.isfinite(results[f"reward_mae_copy@{h}"])
