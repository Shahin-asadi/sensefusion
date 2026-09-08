import numpy as np
import pandas as pd
import pytest

from sensefusion.cli import example_path
from sensefusion.common import InputError, read_table
from sensefusion.core import analyse, combine, merge_blocks


def small_frame():
    rng = np.random.default_rng(812)
    a, b = rng.normal(size=(2, 18))
    return pd.DataFrame(
        {
            "sample_id": [f"r{i}" for i in range(18)],
            "group": np.repeat([f"g{i}" for i in range(6)], 3),
            "target": 2 * a - b + rng.normal(scale=0.1, size=18),
            "A__x": a,
            "B__x": b,
        }
    )


def test_explicit_id_join_is_order_invariant_and_preserves_missing_block():
    joined = merge_blocks(
        {
            "A": pd.DataFrame({"sample_id": ["b", "a"], "x": [2, 1]}),
            "B": pd.DataFrame({"sample_id": ["a", "c"], "x": [10, 30]}),
        }
    ).set_index("sample_id")
    assert joined.loc["a", "A__x"] == 1 and joined.loc["a", "B__x"] == 10
    assert pd.isna(joined.loc["b", "B__x"])


def test_renormalize_available_weights_and_abstain_when_empty():
    result = combine(np.array([[2.0, 4.0], [np.nan, 4.0], [np.nan, np.nan], [3.0, np.nan]]), [0.0, 1.0])
    np.testing.assert_allclose(result[[0, 1, 3]], [4.0, 4.0, 3.0])
    assert np.isnan(result[2])


def test_missing_labels_and_blocks_produce_honest_coverage():
    frame = small_frame()
    frame.loc[0, "target"] = np.nan
    frame.loc[[1, 2], "A__x"] = np.nan
    frame.loc[3, ["A__x", "B__x"]] = np.nan
    result = analyse(frame)
    predictions = result.tables["predictions"]
    assert predictions[predictions.sample_id.eq("r0")].evaluation.eq("unscored_full_fit").all()
    assert predictions[predictions.sample_id.eq("r3") & predictions.model.eq("late_weighted")].prediction.isna().all()
    weights = result.tables["fitted_choices"]
    np.testing.assert_allclose(weights.groupby("fold").weight.sum(), 1)
    assert (weights.weight >= 0).all()
    splits = result.tables["splits"]
    for _, fold in splits.groupby("fold"):
        assert not set(fold[fold.role.eq("train")].group) & set(fold[fold.role.eq("test")].group)


def test_held_out_target_does_not_change_its_training_choices_or_predictions():
    frame = small_frame()
    first = analyse(frame)
    altered = frame.copy()
    altered.loc[altered.group.eq("g0"), "target"] += 1000
    second = analyse(altered)
    one = first.tables["predictions"].query("group == 'g0'").prediction.to_numpy()
    two = second.tables["predictions"].query("group == 'g0'").prediction.to_numpy()
    np.testing.assert_allclose(one, two, atol=1e-10)


def test_no_response_audit_and_target_not_allowed_as_feature():
    frame = small_frame()
    assert analyse(frame.drop(columns="target")).status == "audit_only"
    with pytest.raises(InputError):
        analyse(frame, block_columns={"bad": ["target"]})


def test_public_examples_cover_two_distinct_datasets():
    for name, expected in [("olive_fusion.csv", 240), ("temperature_bands.csv", 9)]:
        result = analyse(read_table(example_path(name)))
        score = result.tables["scores"]
        assert result.status == "complete"
        assert score.n.min() == expected
        assert np.isfinite(score.rmse).all()
