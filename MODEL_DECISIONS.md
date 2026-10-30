# Model decisions frozen before the final check

These decisions were recorded before computing any September-October model
metrics. The executable configuration is `model_policy.json`. The final-check
manifest records its SHA-256 hash together with the implementation and input
hashes so each reported experiment can be identified.

## Final load predictions

Use `catboost_full_mae` for every final load, including unfamiliar cities. This
candidate had the best pooled development MAE ($140.70 across 19,367 loads),
compared with $146.67 for the market-only variant and $214.45 for the equipment
rate-per-mile baseline. The market-only variant had slightly better development
RMSE ($634.00 versus $635.20), which is retained as a metric tradeoff.

The market-only variant also had a small advantage in the fixed unseen-city
stress test ($170.81 versus $174.35 MAE). That test used one city subset and
reduced training data; it is insufficient evidence for additional city-specific
routing. Use the same full model for all final rows. The changing relationship
of quote signal with rate remains a documented risk, and its quote-time
availability is undocumented. No month-specific signal correction is applied.

## Fixed December chart

Use `catboost_chart_mae`, trained from the same labeled rows but restricted to
pickup, delivery, distance, equipment, weight, and derived date/quality features.
It achieved $161.55 pooled development MAE with all six unavailable inputs
excluded during both fitting and validation. Do not invent future market indices
or quote signals or fill chart coordinates from the final inputs.

The chart is a separate reduced-input prediction path. Its values are not
required to equal the full model's predictions on similar final-set loads.

## Fixed training and evaluation settings

- CatBoost MAE objective on raw dollar rates, 500 iterations, depth 6, learning
  rate 0.05, L2 leaf regularization 5, seed 42, and four CPU threads.
- Use the existing stateless features and invalid-weight handling. No imputation
  statistics, encodings, or model parameters are fitted on validation rows.
- Keep every labeled load; no outlier deletion, learned target clipping,
  post-hoc calibration, ensembling, or early stopping against final-check labels.
- Retain the existing fixed $0.01 lower prediction bound.
- For the final check, fit on January-August (38,477 rows) and evaluate on
  September-October (9,523 rows). Score the selected main model, selected chart
  path, and the predeclared equipment baseline. Report MAE, RMSE, R-squared,
  and relevant segments without selecting a new model from these scores.
- After the check, refit the same selected configurations on all 48,000 labeled
  rows for the November-December submission. These refits do not turn the
  September-October results into scores for the fully refitted models.

The employer's hidden metric is unspecified. MAE was chosen for development
selection, and RMSE is reported to expose sensitivity to large errors. A single
partial calendar year cannot establish reliable annual seasonal behavior.

Re-running the frozen experiment is a reproducibility check, not a new tuning
round. If later decisions use its errors to change the model, the final-check
window must be described as additional development data rather than untouched
evaluation data.
