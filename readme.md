# Freight Rate Prediction Challenge

## Implementation progress

The supplied data files are in this directory; commands below use their actual
filenames. See [the assessment brief](freight-rate-ml-assessment-1-.pdf).

- Environment and dependency setup: [SETUP.md](SETUP.md)
- Requirements and implementation sequence: [TASK_OVERVIEW.md](TASK_OVERVIEW.md)
- Generated data-quality analysis: [reports/eda/exploration.md](reports/eda/exploration.md)
- Development model comparison: [reports/benchmark/benchmark.md](reports/benchmark/benchmark.md)
- Manual Git checkpoints: [GIT_WORKFLOW.md](GIT_WORKFLOW.md)

From the project folder, run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m freight.explore
python -m unittest discover -s tests -v
python -m freight.benchmark
```

Activation is optional: replace `python` with `.\.venv\Scripts\python.exe`.
The analysis command writes reports to `reports/eda/` without changing the input
CSVs. The benchmark compares eight candidates on two temporal development folds,
writes `reports/benchmark/`, and leaves September-October unscored. It normally
takes a few minutes on a laptop. Large row-level diagnostics go in ignored
`artifacts/benchmark/`. Location holdouts and final model decisions come next;
submission predictions have not been generated.

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
