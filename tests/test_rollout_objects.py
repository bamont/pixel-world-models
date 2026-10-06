import math

import numpy as np

from pixelwm.eval import open_loop_eval
from pixelwm.models import RSSM, RSSMConfig


def test_object_metrics_keys_and_ranges(small_buffer):
    model = RSSM(2, RSSMConfig(deter=16, groups=2, classes=4, hidden=16, cnn_depth=8))
    results = open_loop_eval(
        model, small_buffer, horizons=(1, 4), context=3, num_sequences=6, object_metrics=True
    )
    for name in ("agent", "goal"):
        for h in (1, 4):
            for key in (f"{name}_err@{h}", f"{name}_found@{h}", f"{name}_err_copy@{h}"):
                assert key in results
            found = results[f"{name}_found@{h}"]
            assert math.isnan(found) or 0.0 <= found <= 1.0
    # The copy baseline always "finds" the objects it copies from a real frame.
    assert np.isfinite(results["agent_err_copy@4"])
    assert results["agent_err_copy@4"] >= results["agent_err_copy@1"] - 5.0


def test_object_metrics_are_off_by_default(small_buffer):
    model = RSSM(2, RSSMConfig(deter=16, groups=2, classes=4, hidden=16, cnn_depth=8))
    results = open_loop_eval(model, small_buffer, horizons=(1,), context=3, num_sequences=4)
    assert not any(k.startswith(("agent_", "goal_")) for k in results)
