"""Closeout regressions: report selection, cohort traces and export boundaries."""

import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sensefusion import core, templates
from sensefusion.presentation import overview_content, primary_figures, select_view
from sensefusion.reporting import files_for_result, summary_files


def test_summary_real_paragraphs_and_traceable_view():
    result = core.analyse(templates.starter_frame(), **templates.schema()["config"])
    with patch.object(core, "analyse", side_effect=AssertionError("Viewing must not refit")):
        output = summary_files(result)
        text = output["summary.txt"].decode("utf-8")
        assert "\n\n" in text and r"\n" not in text
        view = json.loads(output["summary_view.json"])
        cohort = pd.read_csv(io.BytesIO(output["summary_cohort.csv"]))
        assert cohort.included_in_primary_score_and_figure.sum() == view["n_primary_pairs"]


def test_stale_selection_resolves_and_zip_summary_matches():
    result = core.analyse(templates.starter_frame(), **templates.schema()["config"])
    default = select_view(result)
    stale = select_view(result, method="removed", domain="removed", response="removed", budget=9999)
    for key in ["method", "domain", "response", "budget", "scope"]:
        assert stale[key] == default[key]
    selection = {k: default[k] for k in ["method", "domain", "response", "budget"]}
    full = files_for_result(result, formats=("png",), png_dpi=70, **selection)
    assert json.loads(full["summary_view.json"])["scope"] == default["scope"]
    assert full["summary.txt"] == summary_files(result)["summary.txt"]


@pytest.mark.parametrize(
    "mode,n", [("one", 8), ("missing", 8), ("all", 8), ("one", 4), ("missing", 4), ("all", 4), ("all", 2)]
)
def test_default_view_eligible_when_modality_unavailable(mode, n):
    frame = templates.starter_frame().head(n)
    config = templates.schema()["config"].copy()
    if mode == "one":
        config["block_columns"] = {"A": ["Sensor A"]}
    else:
        frame["Sensor B"] = np.nan
        if mode == "all":
            frame["Sensor A"] = np.nan
    result = core.analyse(frame, **config)
    view = select_view(result)
    assert view["method"] in result.settings["common_comparison"]["methods"]
    assert not view["method"].startswith(("late_", "early_"))
    assert int(dict(overview_content(result, view)["cards"])["Usable pairs"]) > 0
    assert "Fusion is unavailable" in overview_content(result, view)["caveat"]


@pytest.mark.parametrize("missing,unlabelled", [(0, False), (20, False), (20, True)])
def test_primary_cohort_membership_matches_metric(missing, unlabelled):
    from sensefusion.cli import example_path

    frame = pd.read_csv(example_path("olive_fusion.csv"))
    frame.loc[frame.index[:missing], frame.columns.str.startswith("UV__")] = np.nan
    if unlabelled:
        frame.loc[239, "target"] = np.nan
    result = core.analyse(frame)
    view = select_view(result)
    expected = set(frame.loc[frame.index >= missing, "sample_id"].astype(str))
    if unlabelled:
        expected.remove(str(frame.loc[239, "sample_id"]))
    actual = set(view["predictions"].sample_id.astype(str))
    assert actual == expected
    assert int(view["scores"].iloc[0]["n"]) == len(expected)
    assert np.sqrt(np.mean(view["predictions"].residual ** 2)) == pytest.approx(view["scores"].iloc[0].rmse)
    output = summary_files(result)
    trace = pd.read_csv(io.BytesIO(output["summary_cohort.csv"]))
    assert set(trace.loc[trace.included_in_primary_score_and_figure, "sample_id"].astype(str)) == expected


def test_no_common_cohort_is_standalone_not_fusion_comparison():
    frame = templates.starter_frame().head(8)
    frame.loc[:3, "Sensor A"] = np.nan
    frame.loc[4:, "Sensor B"] = np.nan
    result = core.analyse(frame, **templates.schema()["config"])
    assert result.settings["common_comparison"]["n_rows"] == 0
    view = select_view(result)
    assert view["scope"] == "standalone_available_pairs"
    assert view["comparison"].empty
    assert view["n_primary_pairs"] > 0
    assert "standalone" in overview_content(result, view)["caveat"]


def test_one_unit_audit_has_no_usable_prediction():
    result = core.analyse(templates.starter_frame().head(1), **templates.schema()["config"])
    view = select_view(result)
    assert view["predictions"].empty
    assert primary_figures(result, view) == []
