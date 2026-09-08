# Methods and interpretation

## Question and unit of validation

Does combining measurement blocks improve prediction of a continuous response, and how often can each method predict when a block is missing? A block is a set of columns from one device or a declared feature family. Block membership is a scientific input, not something inferred from correlations. Rows from the same physical specimen or batch stay together in every validation split. The supplied group must match the population to which the result will be applied [P01–P06, P13–P14].

The program accepts one to six blocks. With one usable block, only the supported single-block model and mean baseline are eligible comparisons. Fusion-labelled fallback rows remain in detailed predictions and are excluded from the comparison. The `merge_blocks` Python function joins separately measured tables on explicit unique IDs. It never assumes that equal row positions imply the same specimen. The guided interface accepts one already joined CSV or XLSX.

## Fitting and evaluation

The outer evaluation uses at most five GroupKFold folds. Each labelled observation appears in one outer test fold. Within each applicable training set, PLS component counts 1, 2 and 3 are compared by at most three grouped folds; ties prefer the smaller count. The effective count cannot exceed the training matrix rank, number of features, or training observations minus one. With fewer than three training groups the fixed choice is one component. A constant response or rank-zero predictor fit returns the training response mean [P08, P11–P12].

Every training fit drops columns having no training measurements, estimates column medians for remaining missing cells, subtracts training column means, and divides each block by the square root of its total training variance. Thus a nonconstant block has unit summed feature variance. The same learned transformation is used for held-out rows; no held-out response or predictor distribution is fitted during preprocessing [P06]. Fitting itself weights observations equally; the tuning score gives physical groups equal weight.

The comparison contains a PLS model for each block, early fusion by concatenating transformed blocks, equal averaging of block predictions, inverse-group-RMSE averaging, and constrained weighted averaging. Weighted fusion learns nonnegative weights summing to one from an additional set of grouped out-of-fold predictions within the outer training set. Base-model tuning is repeated within those inner training subsets. The weights minimize group-weighted squared prediction error. If there are too few complete out-of-fold predictions, equal weights are used and the reason is recorded [P09–P10]. A training-mean predictor supplies a simple reference.

## Incomplete observations

Blank cells inside an available block use training medians. A single-block model abstains if that entire block is absent. Late fusion renormalizes weights over available block predictions; if their learned weights are all zero, it uses their equal mean. With no available block, it abstains. Early fusion imputes absent blocks and may be less reliable in that setting. This is a defined fallback, not a model of the missingness mechanism. Selective or concentration-dependent missingness can bias the comparison.

Missing responses are never estimated for training or evaluation. When enough labelled groups exist, rows with no response receive unscored predictions from models fitted on all labelled data. The nested route requires at least six labelled observations in four evaluation groups. Smaller inputs follow the capability rules below; no universal minimum is imposed.

One to six measurement blocks can be mapped without renaming columns. Two evaluation groups retain a held-out training-mean reference. With at least three groups, variable training folds can fit fixed one-component PLS models; the small-data route does not train nested fusion weights. The existing nested route remains available when supported. Methods without usable predictions and fusion with fewer than two usable blocks are excluded from the common comparison by availability, never by test error. All other predictions, coverage and exclusions remain in the export. A constant-data mean fallback is identified in the fitted-choice table.

## What the scores mean

Residual = prediction − reference. RMSE is the square root of the mean squared residual, MAE is the mean absolute residual, and bias is the mean signed residual. Group RMSE first averages squared errors within groups and then across groups. R² is 1 − SSE/SST and can be negative; it is unavailable for a constant reference. Coverage is the fraction of labelled rows receiving a prediction.

`all_available_rows` preserves each method's coverage; `common_scored_rows` compares methods on the same rows. Compare common-row errors together with available-row coverage. A method that predicts fewer difficult cases can otherwise appear better. A displayed lowest outer-CV error is a descriptive comparison. Selecting that winner and quoting the same score as an independent evaluation introduces another selection step [P11]. Weights are predictive coefficients, not causal sensor importance.

The oil example holds out oil bottle groups, not future ageing dates. It predicts controlled thermal ageing duration at 60 °C, not shelf life. The temperature example splits one device's wavelengths into two fixed bands and contains only nine formulations. It is not a two-instrument experiment. Exact preparation and source limitations are in `DATA_DICTIONARY.md` and `examples/*preparation.json`. No universal benefit of fusion is assumed.

## Inverse-error fusion and diagnostic extensions

Inverse-error weights use each block's group RMSE on the shared nested out-of-fold training cohort. Nonzero errors are inverted and normalized; exact-zero-error blocks share all weight, and the absence of an estimable cohort uses an explicit equal-weight fallback. Test references never select weights. Descriptive per-group and range summaries are computed after fitting; they do not change the evaluation. Numerical response and block normalization avoid fixed absolute cutoffs that would remove a model or R² merely because units changed.


## Default view and reporting population

The view uses eligible methods with a nonempty common score, preferring equal fusion, then single blocks in recorded order, then the training mean. It never chooses the smallest observed error. If the common intersection is empty, an eligible available-pair result is labelled standalone and no family comparison chart is shown. Excluded methods and reasons remain in `common_comparison` and complete prediction tables.

`predictions.in_common_cohort` is recorded by the numerical core at scoring time. The primary scatter and headline use exactly that membership. `summary_cohort.csv` retains identities and inclusion flags for every row of the selected method, while `summary_view.json` records the resolved method, scope and both primary/available pair counts. Unknown references are excluded from both score and primary scatter.
