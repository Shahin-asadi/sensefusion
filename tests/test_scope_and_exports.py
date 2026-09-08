import io
import json

import numpy as np
import pandas as pd
import pytest

from sensefusion import core
from sensefusion.cli import example_config, example_context, example_path
from sensefusion.common import InputError, read_table
from sensefusion.reporting import files_for_result


def test_resolved_configuration_and_portable_public_attribution():
    path = example_path()
    data = read_table(path)
    a = core.analyse(data, **(example_config(path.name) | example_context(path.name)))
    files = files_for_result(a)
    config = json.loads(files["config.json"])
    b = core.analyse(data, **config)
    assert a.settings["input_sha256"] == b.settings["input_sha256"]
    assert json.loads(files["run.json"])["software_version"] == "0.5.2"
    assert json.loads(files["provenance.json"])["original_source"]["license"] == "CC-BY-4.0"
    assert b.settings["provenance"]["kind"] == "bundled_example"
    assert "created_utc" not in config
    for key, table in a.tables.items():
        pd.testing.assert_frame_equal(table, b.tables[key], check_exact=False, rtol=1e-7, atol=1e-9)


def test_uploaded_data_does_not_inherit_bundled_citation():
    path = example_path()
    data = read_table(path)
    result = core.analyse(data, **example_config(path.name))
    assert result.settings["provenance"]["kind"] == "user_provided"
    assert "original_source" not in result.settings["provenance"]


def test_nonobject_cli_config_returns_clear_failure(tmp_path, capsys):
    from sensefusion.cli import main

    file = tmp_path / "bad.json"
    file.write_text("[1,2]")
    assert main(["demo", "--config", str(file), "--output", str(tmp_path / "out")]) == 2
    assert "top level must be an object" in capsys.readouterr().err


def test_duplicate_xlsx_headers_are_rejected():
    frame = pd.DataFrame([["id", "x", "x"], ["001", 1, 2]])
    source = io.BytesIO()
    frame.to_excel(source, index=False, header=False)
    source.seek(0)
    with pytest.raises(InputError, match="unique"):
        read_table(source, "duplicate.xlsx")


def test_html_escapes_untrusted_labels():
    from sensefusion.common import Result
    from sensefusion.reporting import html_report

    result = Result("<script>alert(1)</script>", notes=["<img src=x onerror=alert(1)>"])
    html = html_report(result, "<bad>", False)
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_single_block_and_empty_all_blocks_are_not_fusion_evidence():
    data = read_table(example_path("temperature_bands.csv"))
    for c in data:
        if "__" in c:
            data[c] = np.nan
    r = core.analyse(data)
    assert r.status == "partial"
    common = r.tables["scores"].query("cohort == 'common_scored_rows'")
    assert common.loc[~common.model.eq("training_mean"), "n"].eq(0).all()
    assert common.loc[common.model.eq("training_mean"), "n"].gt(0).all()
    out = files_for_result(r)
    assert "figure.png" in out
    assert r.settings["common_comparison"]["methods"] == ["training_mean"]


def test_mixed_numeric_and_text_ids_pair_by_normalized_key():
    from sensefusion.core import merge_blocks

    r = merge_blocks(
        {
            "A": pd.DataFrame({"sample_id": [1, "NA"], "v": [1, 2]}),
            "B": pd.DataFrame({"sample_id": ["NA", "1"], "v": [3, 4]}),
        }
    )
    assert len(r) == 2 and r.set_index("sample_id").loc["1", "B__v"] == 4
