"""Grouped, nested comparisons of early and late spectroscopic fusion."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import (
    BlockPLS,
    InputError,
    Result,
    audit,
    fingerprint,
    folds,
    identifiers,
    metrics,
    numbers,
    subset,
    tune_pls,
)
from .contract import analysis_contract


def merge_blocks(tables, id_col="sample_id"):
    """Outer join on explicit IDs, preserving absent modalities as missing cells."""
    if not isinstance(tables, dict) or not tables:
        raise InputError("tables: provide one or more named tables with explicit observation IDs.")
    combined = None
    audit_rows = []
    all_keys = []
    for name, table in tables.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(table, pd.DataFrame):
            raise InputError("tables: use non-blank block names mapped to data tables.")
        ids, _ = identifiers(table, id_col)
        table = table.copy()
        table[id_col] = ids
        all_keys.append(set(ids))
        audit_rows.append({"block": name, "rows": len(ids)})
        value_columns = [c for c in table if c != id_col]
        renamed = table.rename(columns={c: f"{name}__{c}" for c in value_columns})
        combined = (
            renamed if combined is None else combined.merge(renamed, on=id_col, how="outer", validate="one_to_one")
        )
    combined.attrs["pairing_audit"] = {
        "rows": len(combined),
        "blocks": audit_rows,
        "matched_all_blocks": len(set.intersection(*all_keys)),
        "normalization": "IDs converted to strings and surrounding whitespace removed before one-to-one joining; leading zeros and literal NA retained",
    }
    return combined


def available(block):
    return np.isfinite(block).any(axis=1)


def combine(predictions, weights):
    valid = np.isfinite(predictions)
    effective = valid * np.asarray(weights)[None, :]
    denominator = effective.sum(axis=1)
    combined = np.full(len(predictions), np.nan)
    supported = denominator > 1e-12
    combined[supported] = (np.nan_to_num(predictions[supported]) * effective[supported]).sum(axis=1) / denominator[
        supported
    ]
    # If all available blocks received zero trained weight, report their equal mean.
    fallback = ~supported & valid.any(axis=1)
    if fallback.any():
        combined[fallback] = np.nansum(predictions[fallback], axis=1) / valid[fallback].sum(axis=1)
    return combined


def train_single(block, y, groups):
    keep = available(block)
    if keep.sum() < 3 or len(np.unique(groups[keep])) < 2:
        return None, None
    component = tune_pls([block[keep]], y[keep], groups[keep])
    fitted = BlockPLS(component).fit([block[keep]], y[keep])
    return fitted, fitted.n_components_


def predict_single(fitted, block):
    prediction = np.full(len(block), np.nan)
    keep = available(block)
    if fitted is not None and keep.any():
        prediction[keep] = fitted.predict([block[keep]])
    return prediction


def train_ensemble(blocks, y, groups):
    models, counts = zip(*(train_single(b, y, groups) for b in blocks))
    oof = np.full((len(y), len(blocks)), np.nan)
    if len(np.unique(groups)) >= 4:
        for train, test in folds(groups, 3):
            for j, block in enumerate(blocks):
                fitted, _ = train_single(block[train], y[train], groups[train])
                oof[test, j] = predict_single(fitted, block[test])
    common = np.isfinite(oof).all(axis=1)
    weights = np.full(len(blocks), 1 / len(blocks))
    inverse_weights = weights.copy()
    inverse_reason = "equal weights: insufficient shared out-of-fold training predictions"
    reason = "equal weights: insufficient complete grouped out-of-fold training predictions"
    if common.sum() >= max(6, 2 * len(blocks)) and len(np.unique(groups[common])) >= 3:
        rmses = np.array([metrics(y[common], oof[common, j], groups[common])["group_rmse"] for j in range(len(blocks))])
        if np.any(rmses == 0):
            inverse_weights = (rmses == 0).astype(float)
        else:
            inverse_weights = rmses.min() / rmses
        inverse_weights /= inverse_weights.sum()
        inverse_reason = "normalized inverse grouped RMSE from shared nested out-of-fold training predictions; exact-zero errors share all weight"
        counts_by_group = pd.Series(groups[common]).value_counts()
        observation_weight = np.array([1 / counts_by_group[g] for g in groups[common]])
        scale = float(np.max(np.abs(y[common] - y[common].mean()))) or 1.0
        optimum = minimize(
            lambda w: float(np.average(((oof[common] @ w - y[common]) / scale) ** 2, weights=observation_weight)),
            weights,
            method="SLSQP",
            bounds=[(0, 1)] * len(blocks),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if optimum.success:
            weights = np.clip(optimum.x, 0, 1)
            weights /= weights.sum()
            reason = "nonnegative sum-to-one weights learned from nested grouped out-of-fold training predictions"
    early_keep = np.logical_or.reduce([available(b) for b in blocks])
    early = None
    early_count = None
    if early_keep.sum() >= 3 and len(np.unique(groups[early_keep])) >= 2:
        early_count = tune_pls(subset(blocks, early_keep), y[early_keep], groups[early_keep])
        early = BlockPLS(early_count).fit(subset(blocks, early_keep), y[early_keep])
        early_count = early.n_components_
    return models, weights, counts, early, early_count, reason, inverse_weights, inverse_reason


def predict_ensemble(fitted, blocks):
    models, weights, _, early, _, _, inverse_weights, _ = fitted
    singles = np.column_stack([predict_single(model, block) for model, block in zip(models, blocks)])
    any_data = np.logical_or.reduce([available(b) for b in blocks])
    early_prediction = np.full(len(any_data), np.nan)
    if early is not None and any_data.any():
        early_prediction[any_data] = early.predict(subset(blocks, any_data))
    return (
        singles,
        early_prediction,
        combine(singles, np.ones(len(blocks))),
        combine(singles, weights),
        combine(singles, inverse_weights),
    )


def small_evaluation(result, frame, y, labelled, ids, groups, evaluation_groups, blocks, block_columns):
    """Prespecified mean and fixed-one-component routes; no low-N tuning or learned weights."""
    indices = np.flatnonzero(labelled)
    n_units = len(np.unique(evaluation_groups[labelled]))
    result.status = "partial"
    result.settings["capability_details"] = [
        {
            "capability": "nested_fusion",
            "available": False,
            "level": "tuning",
            "reason": "Fewer than six labelled rows or four evaluation groups; no component tuning or learned fusion is attempted.",
        }
    ]
    result.notes.append(
        "Small-data evaluation is exploratory. The prespecified training mean remains available; fixed one-component single blocks require two varying training units. No learned fusion weights or tuned winner are claimed."
    )
    if n_units < 2:
        result.status = "audit_only"
        result.notes.append(
            "Only one evaluation unit is available. Input and block descriptions remain available; held-out performance cannot be estimated."
        )
        return result
    names = [f"single_{name}_fixed1" for name in block_columns] if n_units >= 3 else []
    if len(blocks) >= 2 and n_units >= 3:
        names += ["late_equal_fixed1"]
    names += ["training_mean"]
    predictions = np.full((len(frame), len(names)), np.nan)
    fold_ids = np.full(len(frame), -1)
    splits, choices = [], []
    for fold_id, (tr, te) in enumerate(folds(evaluation_groups[labelled], 5)):
        train, test = indices[tr], indices[te]
        singles = []
        if n_units >= 3:
            for name, block in zip(block_columns, blocks):
                usable = available(block[train])
                model = None
                reason = "fewer than two independent training units with measurements"
                if usable.sum() >= 2 and len(np.unique(evaluation_groups[train][usable])) >= 2:
                    candidate = BlockPLS(1).fit([block[train][usable]], y[train][usable])
                    if candidate.n_components_ > 0:
                        model = candidate
                        reason = "prespecified one-component PLS; no tuning"
                    else:
                        reason = "training response or predictors have no usable variation; PLS unavailable"
                single = predict_single(model, block[test])
                singles.append(single)
                choices.append(
                    {"fold": fold_id, "block": name, "components": 1 if model is not None else None, "status": reason}
                )
            columns = singles.copy()
            if len(blocks) >= 2:
                columns.append(combine(np.column_stack(singles), np.ones(len(blocks))))
        else:
            columns = []
        columns.append(np.full(len(test), y[train].mean()))
        predictions[test] = np.column_stack(columns)
        fold_ids[test] = fold_id
        for role, index in [("train", train), ("test", test)]:
            splits.extend(
                {
                    "fold": fold_id,
                    "role": role,
                    "sample_id": ids[i],
                    "group": groups[i],
                    "evaluation_group": evaluation_groups[i],
                }
                for i in index
            )
    usable_blocks = sum(
        np.isfinite(predictions[labelled, j]).any() for j, name in enumerate(names) if name.startswith("single_")
    )
    family = [
        j
        for j, name in enumerate(names)
        if np.isfinite(predictions[labelled, j]).any()
        and (usable_blocks >= 2 or name.startswith("single_") or name == "training_mean")
    ]
    common = labelled & np.isfinite(predictions[:, family]).all(axis=1)
    result.settings["common_comparison"] = {
        "n_rows": int(common.sum()),
        "n_groups": len(np.unique(groups[common])),
        "methods": [names[j] for j in family],
        "excluded_methods": [names[j] for j in range(len(names)) if j not in family],
        "selection_rule": "Usable held-out predictions; fusion requires two usable blocks; never select by error.",
        "status": "exploratory",
    }
    scores = []
    for j, name in enumerate(names):
        for cohort, keep in [
            ("all_available_rows", labelled),
            ("common_scored_rows", common if j in family else np.zeros(len(frame), dtype=bool)),
        ]:
            scores.append({"model": name, "cohort": cohort, **metrics(y[keep], predictions[keep, j], groups[keep])})
    result.tables.update(
        predictions=pd.DataFrame(
            [
                {
                    "sample_id": ids[i],
                    "group": groups[i],
                    "evaluation_group": evaluation_groups[i],
                    "fold": fold_ids[i],
                    "model": name,
                    "reference": y[i],
                    "prediction": predictions[i, j],
                    "in_common_cohort": bool(common[i] and j in family),
                    "residual": predictions[i, j] - y[i],
                    "evaluation": "exploratory_held_out" if labelled[i] else "unscored_no_small_data_fit",
                }
                for j, name in enumerate(names)
                for i in range(len(frame))
            ]
        ),
        scores=pd.DataFrame(scores),
        fitted_choices=pd.DataFrame(choices),
        splits=pd.DataFrame(splits),
    )
    result.settings.update(
        n_rows=len(frame),
        n_labelled=int(labelled.sum()),
        n_groups=len(np.unique(groups[labelled])),
        evaluation_groups=n_units,
    )
    return result


@analysis_contract
def analyse(frame, target="target", id_col="sample_id", group_col="group", block_columns=None, cv_group_col=None):
    ids, groups = identifiers(frame, id_col, group_col)
    _, evaluation_groups = identifiers(frame, id_col, cv_group_col or group_col)
    if cv_group_col and group_col:
        membership = pd.DataFrame({"specimen": groups, "validation_group": evaluation_groups})
        if membership.groupby("specimen").validation_group.nunique().gt(1).any():
            raise InputError(
                "A physical specimen appears in multiple validation groups. Keep every reading of one specimen in the same held-out group."
            )

    if block_columns is None:
        block_columns = {}
        for column in frame:
            if "__" in column and column not in {target, id_col, group_col, cv_group_col}:
                block_columns.setdefault(column.split("__", 1)[0], []).append(column)
    if not 1 <= len(block_columns) <= 6 or any(not cols for cols in block_columns.values()):
        raise InputError(
            "Choose one to six non-empty measurement blocks. The example uses prefixes such as FL__ and UV__."
        )
    features = [c for cols in block_columns.values() for c in cols]
    if len(features) != len(set(features)) or set(features) & {target, id_col, group_col, cv_group_col}:
        raise InputError("Each feature belongs to one block; response and identifier columns cannot be model inputs.")
    blocks = [numbers(frame, cols).to_numpy() for cols in block_columns.values()]
    result = Result(
        "SenseFusion — grouped fusion comparison",
        {"input_audit": audit(frame)},
        settings={
            "input_sha256": fingerprint(frame),
            "target": target,
            "id": id_col,
            "group": group_col,
            "blocks": block_columns,
            "components": [1, 2, 3],
            "outer_max_folds": 5,
            "inner_max_folds": 3,
        },
    )
    result.notes.extend(
        [
            "All transformations and component choices are fitted within training folds. Stack weights use additional out-of-fold training predictions.",
            "Errors have response units; signed residual = prediction minus reference. Group RMSE gives each physical group equal weight.",
            "Compare methods on common_scored_rows; all_available_rows also reports prediction coverage. A displayed best score is descriptive, not an independently validated winner.",
            "Blank cells use training medians. Single-block models abstain on wholly absent blocks; late fusion renormalizes available weights. Early fusion imputes absent blocks and can be less reliable.",
        ]
    )
    if not group_col:
        result.notes.append(
            "Each row is treated as independent. If repeated specimens or batches exist, supply their group IDs and rerun."
        )
    if target not in frame:
        result.status = "audit_only"
        result.notes.append(
            "No reference column: the quality audit is available; supervised modelling and accuracy require measured responses."
        )
        return result
    y = numbers(frame, [target])[target].to_numpy()
    labelled = np.isfinite(y)
    if labelled.sum() < 6 or len(np.unique(evaluation_groups[labelled])) < 4:
        return small_evaluation(result, frame, y, labelled, ids, groups, evaluation_groups, blocks, block_columns)
    labelled_indices = np.flatnonzero(labelled)
    names = [f"single_{b}" for b in block_columns] + [
        "early_pls",
        "late_equal",
        "late_weighted",
        "late_inverse_rmse",
        "training_mean",
    ]
    prediction = np.full((len(frame), len(names)), np.nan)
    fold_ids = np.full(len(frame), -1)
    tuning, splits = [], []
    for fold_id, (train_local, test_local) in enumerate(folds(evaluation_groups[labelled], 5)):
        train, test = labelled_indices[train_local], labelled_indices[test_local]
        fitted = train_ensemble(subset(blocks, train), y[train], evaluation_groups[train])
        singles, early, equal, weighted, inverse = predict_ensemble(fitted, subset(blocks, test))
        prediction[test] = np.column_stack(
            [singles, early, equal, weighted, inverse, np.full(len(test), y[train].mean())]
        )
        fold_ids[test] = fold_id
        _, weights, counts, _, early_count, reason, inverse_weights, inverse_reason = fitted
        for name, weight, count, inverse_weight in zip(block_columns, weights, counts, inverse_weights):
            tuning.append(
                {
                    "fold": fold_id,
                    "block": name,
                    "components": count,
                    "estimator": "training_mean_fallback"
                    if count == 0
                    else "PLS"
                    if count is not None
                    else "unavailable",
                    "weight": weight,
                    "weight_method": reason,
                    "inverse_rmse_weight": inverse_weight,
                    "inverse_rmse_weight_method": inverse_reason,
                    "early_components": early_count,
                    "early_estimator": "training_mean_fallback"
                    if early_count == 0
                    else "PLS"
                    if early_count is not None
                    else "unavailable",
                }
            )
        for role, index in [("train", train), ("test", test)]:
            splits.extend(
                {
                    "fold": fold_id,
                    "role": role,
                    "sample_id": ids[i],
                    "group": groups[i],
                    "evaluation_group": evaluation_groups[i],
                }
                for i in index
            )
    if (~labelled).any():
        full_fit = train_ensemble(subset(blocks, labelled), y[labelled], evaluation_groups[labelled])
        singles, early, equal, weighted, inverse = predict_ensemble(full_fit, subset(blocks, ~labelled))
        prediction[~labelled] = np.column_stack(
            [singles, early, equal, weighted, inverse, np.full((~labelled).sum(), y[labelled].mean())]
        )
        result.notes.append(
            "Rows without a reference receive predictions from all labelled training rows; these predictions are not scored or independently validated."
        )
    long = []
    for j, name in enumerate(names):
        long.extend(
            {
                "sample_id": ids[i],
                "group": groups[i],
                "fold": fold_ids[i],
                "model": name,
                "reference": y[i],
                "prediction": prediction[i, j],
                "residual": prediction[i, j] - y[i],
                "evaluation": "out_of_fold" if labelled[i] else "unscored_full_fit",
            }
            for i in range(len(frame))
        )
    usable_block_count = sum(np.isfinite(prediction[labelled, j]).any() for j in range(len(blocks)))
    family = [
        j
        for j, name in enumerate(names)
        if np.isfinite(prediction[labelled, j]).any()
        and (usable_block_count >= 2 or name.startswith("single_") or name == "training_mean")
    ]
    common = labelled & np.isfinite(prediction[:, family]).all(axis=1)
    if usable_block_count < 2:
        result.status = "partial"
        result.notes.append(
            "Fewer than two usable measurement blocks: only available single-block models and the mean reference form the comparison. This is not evidence for fusion."
        )
    result.settings["common_comparison"] = {
        "n_rows": int(common.sum()),
        "n_groups": len(np.unique(groups[common])),
        "methods": [names[j] for j in family],
        "excluded_methods": [name for j, name in enumerate(names) if j not in family],
        "selection_rule": "Exclude methods without any held-out predictions and fusion when fewer than two single blocks are usable; never select by error.",
        "status": "available" if common.any() else "unavailable",
    }
    if not common.any():
        result.status = "partial"
        result.notes.append(
            "The all-method common comparison is unavailable: zero rows and zero physical groups support every method. Available-method predictions remain descriptive; an absent modality is not evidence for fusion."
        )
    elif common.sum() < labelled.sum():
        result.status = "partial"
        result.notes.append(
            f"The common comparison retains {common.sum()} of {labelled.sum()} labelled rows in {len(np.unique(groups[common]))} physical groups; unavailable predictions are excluded explicitly."
        )
    score_rows = []
    for j, name in enumerate(names):
        comparison_keep = common if j in family else np.zeros(len(frame), dtype=bool)
        for cohort, keep in [("all_available_rows", labelled), ("common_scored_rows", comparison_keep)]:
            score_rows.append({"model": name, "cohort": cohort, **metrics(y[keep], prediction[keep, j], groups[keep])})
    result.tables.update(
        predictions=pd.DataFrame(long).assign(
            in_common_cohort=[bool(common[i] and j in family) for j in range(len(names)) for i in range(len(frame))]
        ),
        scores=pd.DataFrame(score_rows),
        fitted_choices=pd.DataFrame(tuning),
        splits=pd.DataFrame(splits),
    )
    result.settings.update(n_rows=len(frame), n_labelled=int(labelled.sum()), n_groups=len(np.unique(groups[labelled])))
    return result
