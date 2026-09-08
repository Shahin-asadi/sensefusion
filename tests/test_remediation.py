import numpy as np
import pandas as pd
import pytest

from sensefusion import core
from sensefusion.common import BlockPLS, InputError, metrics


@pytest.mark.parametrize("scale", [1e-12, 1e-9, 1, 1e6])
def test_metric_units_are_invariant(scale):
    a = metrics([1, 2, 3], [1.1, 1.8, 3.1], ["a", "b", "c"])
    b = metrics(np.array([1, 2, 3]) * scale, np.array([1.1, 1.8, 3.1]) * scale, ["a", "b", "c"])
    assert b["r2"] == pytest.approx(0.97, abs=1e-12)
    for key in ["rmse", "mae", "bias", "group_rmse"]:
        assert b[key] / scale == pytest.approx(a[key], abs=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1e-9, 1e6])
def test_pls_response_and_predictor_units_roundtrip(scale):
    x = np.array([[1.0, 2], [2, 1], [3, 7], [4, 3], [6, 8], [8, 10]])
    y = np.array([1.0, 2, 3, 2, 7, 8])
    a = BlockPLS(2).fit([x], y).predict([x])
    b = BlockPLS(2).fit([x * scale], y * scale).predict([x * scale]) / scale
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


def test_metrics_translation_negative_r2_and_missing_pairs():
    y = np.array([1.0, 2, 3])
    p = np.array([5.0, 6, 7])
    assert metrics(y, p)["r2"] < 0
    assert metrics(y + 100, p + 100)["r2"] == metrics(y, p)["r2"]
    assert metrics([0, 0], [0, 0])["r2"] is None
    assert metrics([np.nan, 1], [1, np.nan])["n"] == 0


def test_api_rejects_unknown_configuration_with_field_name():
    with pytest.raises(InputError, match="unknown_option"):
        core.analyse(pd.DataFrame(), unknown_option=True)


from sensefusion.core import merge_blocks


def small_frame():
    rng = np.random.default_rng(812)
    a, b = rng.normal(size=(2, 18))
    return pd.DataFrame(
        {
            "sample_id": [f"r{i}" for i in range(18)],
            "group": np.repeat(list("abcdef"), 3),
            "target": 2 * a - b,
            "A__x": a,
            "B__x": b,
        }
    )


def test_join_normalizes_both_tables_and_records_pairing():
    joined = merge_blocks(
        {
            "A": pd.DataFrame({"sample_id": [" x", "001", "NA"], "v": [1, 2, 3]}),
            "B": pd.DataFrame({"sample_id": ["NA", "x ", "004"], "v": [4, 5, 6]}),
        }
    )
    assert len(joined) == 4
    assert joined.set_index("sample_id").loc["x", "B__v"] == 5
    assert joined.attrs["pairing_audit"]["rows"] == 4
    with pytest.raises(InputError, match="repeat"):
        merge_blocks({"A": pd.DataFrame({"sample_id": ["x", " x"], "v": [1, 2]})})
    with pytest.raises(InputError, match="tables"):
        merge_blocks({})


def test_entirely_absent_modality_marks_comparison_unavailable():
    frame = small_frame()
    frame["B__x"] = np.nan
    result = core.analyse(frame)
    assert result.status == "partial"
    common = result.tables["scores"].query("cohort == 'common_scored_rows'").set_index("model")
    assert common.loc["single_A", "n"] == len(frame)
    assert common.loc["training_mean", "n"] == len(frame)
    assert common.loc["single_B", "n"] == 0
    assert common.loc["late_equal", "n"] == 0
    assert result.tables["scores"].query("model == 'single_A' and cohort == 'all_available_rows'").n.iloc[0] == 18
    assert result.settings["common_comparison"]["n_groups"] == 6
