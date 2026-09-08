# Changelog

## 0.5.2 — 2026-09-08

- Resolve displayed methods by eligible evidence and align primary scatter membership with scores.
- Preserve candidate factor labels separately from metadata and state agreement counts/estimands precisely.
- Keep PDF footers outside content flow, repair plain-text paragraphs, and consolidate current input/method documentation.
- Record final patch verification separately from the historical 0.5.1 snapshots.


## 0.5.1 — 2026-09-08

Flexible CSV/XLSX/paste import, optional internal row keys, editable artificial starters, capability-specific small-data routes, concise HTML/PDF, deferred diagnostics and full exports. Regression and execution evidence is recorded in VALIDATION.md.

One to six measurement blocks can be mapped without renaming columns. Two evaluation groups retain a held-out training-mean reference. With at least three groups, variable training folds can fit fixed one-component PLS models; the small-data route does not train nested fusion weights. The existing nested route remains available when supported. Methods without usable predictions and fusion with fewer than two usable blocks are excluded from the common comparison by availability, never by test error. All other predictions, coverage and exclusions remain in the export. A constant-data mean fallback is identified in the fitted-choice table.

## 0.1.1 — supplied baseline

Preserved separately. Historical checks do not certify this release.
