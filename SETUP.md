# Project environment

Run commands from `D:\Downloads\Coding\Spotter_ML` in PowerShell.

## Existing environment

The `.venv` directory uses Python 3.12.14 from the Codex bundled runtime.
It does not inherit the runtime's installed packages. Its base interpreter is:

`C:\Users\affan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

The normal `python` and `python3` commands currently resolve to Windows Store
aliases, not a working standalone Python installation. No separate installation
is needed to use this environment. Because it depends on the Codex runtime path,
recreate it if that runtime is moved or removed. For a standalone setup, use
64-bit CPython 3.12 and recreate the environment with that interpreter.

Use the environment directly (activation is optional):

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Or activate it:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation, use the direct interpreter commands above.

If you prefer activation, PowerShell can permit local scripts for the current
terminal process only:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

For Git Bash/MSYS2, use Unix-style paths and shell activation:

```bash
cd /d/Downloads/Coding/Spotter_ML
source .venv/Scripts/activate
python --version
```

After activation, either shell supports:

```text
python -m freight.explore
python -m unittest discover -s tests -v
python -m freight.benchmark
python -m freight.robustness
python -m freight.final_check
python -m freight.submission
```

The benchmark defaults to 500 CatBoost iterations and four CPU threads. Override
with `python -m freight.benchmark --iterations 500 --threads 4` if desired.
Changing the iteration count produces a new experiment, so retain that setting
when comparing results. It does not evaluate the reserved final-check window.

The robustness command also uses 500 iterations and four threads by default.
It needs `artifacts/benchmark/development_predictions.csv`, which is generated
locally by the benchmark and excluded from Git. It verifies source IDs, labels,
hashes, and scores before analyzing residuals. If you change benchmark iterations,
use the same `--iterations` argument for robustness. Both commands leave the
reserved September-October performance check unscored.

`python -m freight.final_check` evaluates the frozen `model_policy.json` on
September-October. It can run independently of the benchmark diagnostics. A
repeat with the same input, implementation, policy, and environment reproduces
the same experiment. It refuses to overwrite an existing final-check report
from a different experiment. `.gitattributes` fixes source/CSV line endings so
Git checkout on Windows does not change input and implementation hashes.

For a different Python version or platform, preserve the original evidence and
write a separate reproduction directory:

```text
python -m freight.final_check --output-dir reports/final_check_reproduction
```

Neither a reproduction nor a new output directory makes these labels an
independent untouched test set. Do not use the scores for another tuning round.

## Generate submission files

With the evaluated data, code, and frozen policy unchanged, run:

```text
python -m freight.submission
```

This fits the main and chart models on all 48,000 labeled loads, saves native
CatBoost files in ignored `artifacts/models/`, verifies reload predictions, and
writes `validation_predictions.csv` and `december_predictions.csv`. It runs the
supplied scorer automatically and writes `scorer_results/candidate_december.png`.
An audit is generated in `reports/submission/`.

To reuse these saved models without retraining:

```text
python -m freight.submission --predict-only
```

The saved bundle must match the frozen policy, input, implementation, and Python
environment in which that bundle was fitted. A fresh clone must run the fitting command first because models
are intentionally excluded from Git. CSV/PNG deliverables and summary audits
are included in Git. No extra dependencies are needed.

Another Python patch version or platform can refit the same evaluated data,
code, and policy without replacing the original evaluation report. The command
records the generation environment and whether it matches the evaluation
environment; original holdout scores still refer to their recorded environment.
Use the lock file for best reproducibility. Identical values across every
platform/library version are not guaranteed. Reusing a saved model bundle in
another environment is rejected; regenerate it with the fitting command.

## Generate the formal report

After generating predictions and passing the audit, run:

```text
python -m freight.report
```

This writes `output/pdf/freight_rate_assessment_report.pdf` using audited metrics
and the exact scorer chart. ReportLab and PDF verification dependencies are
already pinned. The prepared document was rendered and visually reviewed on all
six pages. `LOOM_WALKTHROUGH.md` contains the timed script and recording plan;
the actual camera-on recording and accessible link must be supplied separately.

## Recreate on another computer

With Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` is the unchanged employer-supplied scorer dependency list.
`requirements-project.txt` includes it and adds scientific computing,
scikit-learn baselines, CatBoost, model persistence, and PDF report tools.
CatBoost supplies the selected main and reduced-input chart models.
`requirements-lock.txt` records the exact installed package versions.

To resolve the allowed version ranges again instead of reproducing the lock:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-project.txt
```

## Supplied file paths

The supplied README refers to a `data/` directory and underscore-separated
filenames. The actual CSVs are in the project root and some use hyphens. Keep
these original inputs intact and use their actual paths.

After generating the two prediction files, run the unchanged scorer:

```powershell
.\.venv\Scripts\python.exe score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

`december_predictions.csv` should be a completed copy of
`december-chart-inputs.csv`, retaining all seven columns and original input values.
The scorer writes `scorer_results/candidate_december.png`. This command is
also available independently of the submission-generation command.
