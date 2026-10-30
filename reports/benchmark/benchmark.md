# Development benchmark

Run: `python -m freight.benchmark --iterations 500 --threads 4`.
All candidates use fixed parameters and random seed 42. No final-input labels
exist, and September-October model performance has not been evaluated.

## Pooled development results

| model | rows | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| catboost_full_mae | 19367 | 140.7030 | 635.2020 | 0.8253 |
| catboost_market_mae | 19367 | 146.6700 | 634.0013 | 0.8259 |
| catboost_core_mae | 19367 | 160.3659 | 638.1137 | 0.8237 |
| catboost_chart_mae | 19367 | 161.5545 | 638.3546 | 0.8236 |
| catboost_core_rmse | 19367 | 198.0229 | 671.4186 | 0.8048 |
| ridge_core_rate_per_mile | 19367 | 213.1825 | 651.3480 | 0.8163 |
| equipment_rate_per_mile | 19367 | 214.4471 | 658.5362 | 0.8122 |
| median_rate | 19367 | 1152.0660 | 1574.7938 | -0.0738 |

Ranked by MAE in dollars, with RMSE as a secondary comparison because the hidden
metric is unspecified. Scores pool the nonoverlapping May-June and July-August
validation rows; each load has one out-of-time prediction per model. The best
development MAE is **$140.70** from **catboost_full_mae**, a **34.4%**
reduction from the equipment rate-per-mile baseline. This is a development
comparison, not an unbiased final performance estimate.

## Individual temporal backtests

| fold | model | rows | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| development_1 | median_rate | 9696 | 1175.9588 | 1616.1710 | -0.0913 |
| development_1 | equipment_rate_per_mile | 9696 | 206.1843 | 654.1803 | 0.8212 |
| development_1 | ridge_core_rate_per_mile | 9696 | 155.6776 | 642.8722 | 0.8273 |
| development_1 | catboost_core_mae | 9696 | 134.6990 | 642.1095 | 0.8277 |
| development_1 | catboost_market_mae | 9696 | 129.3062 | 639.3107 | 0.8292 |
| development_1 | catboost_full_mae | 9696 | 139.1380 | 643.4172 | 0.8270 |
| development_1 | catboost_core_rmse | 9696 | 143.2610 | 646.2265 | 0.8255 |
| development_1 | catboost_chart_mae | 9696 | 139.9998 | 642.8031 | 0.8274 |
| development_2 | median_rate | 9671 | 1128.1113 | 1532.1883 | -0.0566 |
| development_2 | equipment_rate_per_mile | 9671 | 222.7312 | 662.8747 | 0.8022 |
| development_2 | ridge_core_rate_per_mile | 9671 | 270.8360 | 659.7364 | 0.8041 |
| development_2 | catboost_core_mae | 9671 | 186.0990 | 634.0823 | 0.8190 |
| development_2 | catboost_market_mae | 9671 | 164.0786 | 628.6332 | 0.8221 |
| development_2 | catboost_full_mae | 9671 | 142.2720 | 626.8574 | 0.8231 |
| development_2 | catboost_core_rmse | 9671 | 252.9263 | 695.7608 | 0.7821 |
| development_2 | catboost_chart_mae | 9671 | 183.1650 | 633.8633 | 0.8192 |

Training expands January-April -> May-June and January-June -> July-August.
Model selection uses these two folds only. Imputation, scaling, one-hot encoding,
equipment medians, and all estimator fits use the respective training rows.
There is no early stopping against validation labels and no outlier deletion.

## Candidate definitions

- `median_rate`: one median training rate for every load.
- `equipment_rate_per_mile`: median training rate/mile by equipment, multiplied
  by distance; unseen equipment falls back to the global training median rate/mile.
- `ridge_core_rate_per_mile`: linear regularized rate/mile regression with
  training-fitted median imputation, numeric scaling, and unknown-safe one-hot encoding.
- `catboost_core_mae`: raw-dollar MAE objective; supplied coordinates, cities,
  equipment, distance, cleaned weight/flags, and calendar features; neither signal.
- `catboost_market_mae`: core plus market index and its missingness indicator.
- `catboost_full_mae`: market features plus quote signal.
- `catboost_core_rmse`: same core features, squared-error training objective.
- `catboost_chart_mae`: only December-available features and derived calendar/
  weight flags. No coordinates, market index, or quote signal are required.

CatBoost uses 500 iterations, depth 6, learning rate 0.05, and L2 leaf
regularization 5. All predictions have a fixed $0.01 lower bound to satisfy
positive-rate contracts. No model artifacts are serialized at this stage.

## Diagnostics and limitations

`segment_metrics.csv` reports month, equipment, distance band, weight quality,
market missingness, and city novelty for each model/fold. These time folds have
no unseen cities; a separate labeled location holdout is still required to
assess that risk. The raw out-of-fold predictions are reproducible in ignored
`artifacts/benchmark/development_predictions.csv`.

The quote-signal ablation measures sensitivity to its changing relationship.
The chart candidate is trained and evaluated without all six omitted inputs,
so its scores assess the reduced-input path directly. Ranking one overall model
does not settle which model to use for December chart rows.

Before finalizing: inspect fold consistency and segment errors; run a location
holdout; lock the model decisions; evaluate September-October once; then refit
on all labeled development data and generate submission predictions.
