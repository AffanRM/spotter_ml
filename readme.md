# Freight Rate Prediction Challenge

Predict freight load rates with chronological validation and reproducible
CatBoost pipelines. The reserved September-October evaluation achieved
**$116.19 MAE / $636.24 RMSE** for the main model, versus **$229.10 MAE** for
the equipment baseline. These are internal evaluation scores, not hidden
submission scores. The final refits use all 48,000 labeled rows.

## Reviewer quick start

Install 64-bit Python 3.12, clone this repository, and open a terminal in its root.
Create and activate a virtual environment using [SETUP.md](SETUP.md), then run:

```text
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python -m freight.submission
python -m freight.audit --require-saved-models
```

This generates both prediction CSVs and the supplied scorer's December chart.
No private account, GPU, external data service, or pretrained artifact is needed.
The committed outputs can also be checked immediately with the supplied scorer:

```text
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

## Implementation progress

The supplied data files are in this directory; commands below use their actual
filenames. See [the assessment brief](freight-rate-ml-assessment-1-.pdf).

- Environment and dependency setup: [SETUP.md](SETUP.md)
- Requirements and implementation sequence: [TASK_OVERVIEW.md](TASK_OVERVIEW.md)
- Generated data-quality analysis: [reports/eda/exploration.md](reports/eda/exploration.md)
- Development model comparison: [reports/benchmark/benchmark.md](reports/benchmark/benchmark.md)
- Geographic stress tests and residuals: [reports/robustness/robustness.md](reports/robustness/robustness.md)
- Frozen model decisions: [MODEL_DECISIONS.md](MODEL_DECISIONS.md)
- Reserved-window evaluation: [reports/final_check/final_check.md](reports/final_check/final_check.md)
- Submission generation audit: [reports/submission/submission.md](reports/submission/submission.md)
- Full project review: [PROJECT_REVIEW.md](PROJECT_REVIEW.md)
- Manual Git checkpoints: [GIT_WORKFLOW.md](GIT_WORKFLOW.md)

From the project folder, run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m freight.explore
python -m unittest discover -s tests -v
python -m freight.benchmark
python -m freight.robustness
python -m freight.final_check
python -m freight.submission
```

Activation is optional: replace `python` with `.\.venv\Scripts\python.exe`.
The analysis command writes reports to `reports/eda/` without changing the input
CSVs. The benchmark compares eight candidates on two temporal development folds,
writes `reports/benchmark/`, and leaves September-October unscored. It normally
takes a few minutes on a laptop. Large row-level diagnostics go in ignored
`artifacts/benchmark/`. The robustness command retrains five candidates after
removing eight cities from training, scores seen/unseen locations separately,
and audits the original temporal prediction errors. It writes
`reports/robustness/` and requires the benchmark's local row-level diagnostics;
run the benchmark first on a fresh clone. The selected main/chart models and
parameters are recorded in `model_policy.json`; `freight.final_check` evaluates
those frozen choices on September-October. It reports the predeclared baseline
and both inference paths without selecting a new model from that window.
The submission command refits both frozen models on all 48,000 labeled rows,
checks native-model reload predictions, fills both output CSVs, and invokes the
unchanged scorer. Saved models are generated locally in ignored
`artifacts/models/`. Use `python -m freight.submission --predict-only` to reuse
matching saved models without retraining.

Generated deliverables:

- [12,000 final load predictions](validation_predictions.csv)
- [31 fixed December predictions](december_predictions.csv)
- [Scorer-generated December chart](scorer_results/candidate_december.png)

Both prediction files passed the supplied scorer. The formal report and Loom
recording are the remaining submission deliverables.

In Git Bash/MSYS2, activate with `source .venv/Scripts/activate` instead. The
subsequent `python` commands are the same in either shell.

## What to do

1. Train and validate your model using `train-test.csv`.
2. Predict every load in `validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `validation-predictions-template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `december-chart-inputs.csv` by filling its `predicted_rate` column. Save a completed copy as `december_predictions.csv`.
5. After these prediction files have been generated, run:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
