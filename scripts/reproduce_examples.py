"""Recreate compact examples from versioned public data. See DATA_SOURCES.json."""

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = DEST = Path(".")


def olive():
    records, uv, acquisitions = {}, {}, {}
    with zipfile.ZipFile(DATA / "g6y69g8gwm" / "Raw_data.zip") as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            step = int(re.search(r"Aging Step (\d+)", name)[1])
            rows = list(csv.reader(archive.read(name).decode("utf-8-sig", errors="replace").splitlines()))
            if "/Fluorescence/" in name:
                by_sample = {}
                for col, label in enumerate(rows[0]):
                    match = re.fullmatch(r"([A-X][0-2])_EX_(\d+\.\d+)", label)
                    if not match:
                        continue
                    sample, excitation = match[1], int(float(match[2]))
                    if excitation not in range(340, 621, 40):
                        continue
                    numeric = np.array([[float(row[col]), float(row[col + 1])] for row in rows[2:253]])
                    emission, intensity = numeric.T
                    for centre in range(490, 791, 60):
                        mask = (
                            (abs(emission - centre) <= 20)
                            & (emission > excitation + 20)
                            & (abs(emission - 2 * excitation) > 20)
                        )
                        if mask.sum() < 5:
                            continue
                        by_sample.setdefault(sample, {})[f"FL__ex{excitation}_em{centre}"] = float(
                            intensity[mask].mean()
                        )
                for sample, features in by_sample.items():
                    key = (step, sample)
                    records.setdefault(key, []).append(features)
                    acquisitions.setdefault(key, []).append(name)
            else:
                for row in rows:
                    if row and re.fullmatch(r"[A-X][0-2]", row[0].strip()):
                        key = (step, row[0].strip())
                        if key in uv:
                            raise ValueError(f"Duplicate UV reference: {key}")
                        uv[key] = dict(zip(["UV__K232", "UV__K264", "UV__K268", "UV__K272"], map(float, row[2:6])))
    days = [0, 2, 4, 7, 9, 18, 27, 36, 45, 53]
    output, duplicate_audit = [], []
    for key in sorted(uv):
        step, sample = key
        candidates = pd.DataFrame(records[key])
        if len(candidates) > 1:
            duplicate_audit.append(
                {
                    "id": f"AS{step}_{sample}",
                    "files": acquisitions[key],
                    "max_feature_range": float((candidates.max() - candidates.min()).max()),
                    "action": "arithmetic mean of repeated acquisitions before splitting",
                }
            )
        output.append(
            {
                "sample_id": f"AS{step}_{sample}",
                "group": sample[0],
                "target": days[step],
                **candidates.mean().to_dict(),
                **uv[key],
            }
        )
    frame = pd.DataFrame(output)
    assert len(frame) == 240 and frame.sample_id.is_unique and not frame.isna().any().any()
    frame.to_csv(DEST / "olive_fusion.csv", index=False)
    assay = frame[["sample_id", "group"] + [c for c in frame if c.startswith("FL__")]].copy()
    assay.insert(2, "target", frame["UV__K268"])
    assay.to_csv(DEST / "olive_assay.csv", index=False)
    (DEST / "olive_preparation.json").write_text(
        json.dumps(
            {
                "source_doi": "10.17632/g6y69g8gwm.1",
                "license": "CC-BY-4.0",
                "creators": ["Francesca Venturini", "Silvan Fluri", "Michael Baumgartner"],
                "changes": "240 UV-matched physical vial/ageing-step records. Fixed fluorescence window means; no response-based feature selection. Repeated acquisitions averaged by physical key.",
                "fluorescence": {
                    "excitation_nm": list(range(340, 621, 40)),
                    "emission_centres_nm": list(range(490, 791, 60)),
                    "window_half_width_nm": 20,
                    "min_points": 5,
                    "masks": "emission > excitation + 20; abs(emission - 2*excitation) > 20",
                    "negative_intensities": "retained",
                    "inner_filter_correction": "not performed; undiluted oils",
                },
                "fusion_target": "Controlled thermal ageing duration at 60 C in days; not shelf life or natural age.",
                "assay_target": "Deposited UV K268 reference; UV features excluded from assay inputs.",
                "n_independent_oil_groups": 24,
                "n_rows": 240,
                "duplicate_audit": duplicate_audit,
                "unmatched_fluorescence_keys_excluded": len(set(records) - set(uv)),
                "source_note": "P21 prints an inconsistent delta-K formula and once says 260 rather than 268 nm. No delta-K is calculated; raw UV headers define wavelengths.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def temperature():
    frame = pd.read_csv(DATA / "4p63x5r852" / "Temperature_Variation_Dataset.csv")
    features = [c for c in frame if re.fullmatch(r"\d+(?:\.\d+)?", c)]
    averaged = frame.groupby(["subset", "sample"], sort=True)[features + ["Conc"]].mean().reset_index()
    result = pd.DataFrame(
        {
            "sample_id": averaged["subset"] + "_" + averaged["sample"].astype(str),
            "group": averaged["sample"].astype(str),
            "domain": averaged["subset"],
            "target": averaged["Conc"],
        }
    )
    result = pd.concat([result, averaged[features].rename(columns={f: f"NIR__{f}" for f in features})], axis=1)
    result.to_csv(DEST / "temperature_transfer.csv", index=False)
    bands = result[result.domain.eq("X1")].drop(columns="domain").copy()
    bands = bands.rename(columns={f"NIR__{f}": f"{'LOW' if float(f) < 1300 else 'HIGH'}__{f}" for f in features})
    bands.to_csv(DEST / "temperature_bands.csv", index=False)
    (DEST / "temperature_preparation.json").write_text(
        json.dumps(
            {
                "source_doi": "10.17632/4p63x5r852.1",
                "license": "CC-BY-4.0",
                "creators": ["Ahmed Ramadan", "Nicolas Abatzoglou", "Ryan Gosselin"],
                "changes": "3141 spectra averaged within nine formulation IDs and four acquisition domains, producing 36 rows. No fitted preprocessing at preparation stage.",
                "target": "Acetaminophen % w/w",
                "group": "Formulation ID; all conditions of a held-out formulation excluded from fitting and adaptation.",
                "bands_example": "X1 only, fixed split at 1300 nm. Two wavelength bands of one device, not independent instruments; only nine groups.",
                "limitations": "Fixed measurement order, nine formulations, no independent preparation replication. Public benchmark, not reproduction of the paper's scan-level validation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    import argparse
    import urllib.request

    global DATA, DEST
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Recreate the bundled examples from checksum-verified public measurements."
    )
    parser.add_argument("--raw-dir", type=Path, default=root / "user_data" / "public_raw")
    parser.add_argument("--output", type=Path, default=root / "runs" / "recreated_examples")
    parser.add_argument("--download", action="store_true", help="Fetch missing CC BY 4.0 source files over HTTPS.")
    args = parser.parse_args()
    DATA, DEST = args.raw_dir.resolve(), args.output.resolve()
    if DEST.exists() and any(DEST.iterdir()):
        parser.error("Choose a new or empty output directory; existing files are preserved.")
    sources = json.loads((root / "DATA_SOURCES.json").read_text(encoding="utf-8"))
    for source in sources:
        if source["license"] != "CC-BY-4.0":
            parser.error("This adapter only accepts the reviewed CC BY 4.0 source manifest.")
        for item in source["downloads"]:
            path = DATA / source["repository_id"] / item["file"]
            if not path.exists():
                if not args.download:
                    parser.error(f"Missing {path}. Supply --download or place the original file there.")
                print(f"Downloading {source['doi']} ({source['license']}): {item['file']}")
                request = urllib.request.Request(
                    item["url"], headers={"User-Agent": "Research-example-reproduction/0.1"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    parser.error(f"Download checksum mismatch for {item['file']}; no source file was saved.")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            data = path.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                parser.error(
                    f"Source checksum mismatch for {path}. Check the dataset version; the file was not modified."
                )
    DEST.mkdir(parents=True, exist_ok=True)
    olive()
    temperature()
    manifest = {
        p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in DEST.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Examples recreated in {DEST}. Compare numeric CSV values with examples/; float formatting can vary with library versions."
    )


if __name__ == "__main__":
    main()
