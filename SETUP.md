# Environment and reproducibility

Use a 64-bit Python 3.12 installation. Run commands from the repository root.
The virtual environment and generated model binaries are excluded from Git.

## Create an environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, permit scripts for the current terminal process:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Windows Git Bash:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install and check the pinned dependencies:

```text
python -m pip install -r requirements-lock.txt
python -m pip check
python -m unittest discover -s tests -v
```

`requirements.txt` is the original scorer dependency list.
`requirements-project.txt` declares the additional project dependencies, while
`requirements-lock.txt` pins the exact installed versions used for reproduction.

## Generate and verify predictions

```text
python -m freight.submission
python -m freight.audit --require-saved-models
```

This refits the frozen configurations on all 48,000 labeled loads, saves native
CatBoost files in `artifacts/models/`, verifies reload predictions, writes both
prediction CSVs, and invokes the unchanged supplied scorer. The command creates
`scorer_results/candidate_december.png` and the audit in `reports/submission/`.
A fresh clone needs no pre-existing model files.

To reuse a matching saved bundle without retraining:

```text
python -m freight.submission --predict-only
```

Saved bundles must match their recorded input, implementation, policy and
Python environment. Refit rather than reusing a bundle in a changed environment.
Another Python patch version or platform may refit the same evaluated code and
policy; the generation environment is recorded separately from the original
reserved evaluation. Exact values across every platform are not guaranteed.

## Reproduce analysis

```text
python -m freight.explore
python -m freight.benchmark
python -m freight.robustness
```

Run the benchmark before robustness: it produces the ignored row-level
predictions used for residual diagnostics. Defaults are 500 CatBoost iterations
and four threads. Changing parameters is a different experiment.

The frozen evaluation can be rerun with `python -m freight.final_check` in its
original environment. For another Python version or platform, preserve the
original report and write a separate reproduction:

```text
python -m freight.final_check --output-dir reports/final_check_reproduction
```

The command refuses to overwrite results from a different experiment. Reusing
these labels for tuning would make them additional development data rather
than an independent evaluation. `.gitattributes` preserves source/CSV line
endings across checkouts for the recorded hashes.

## Rebuild the formal report

```text
python -m freight.report
```

The output is `output/pdf/freight_rate_assessment_report.pdf`. It uses audited
metrics and the exact scorer-generated December chart. Required ReportLab and
PDF packages are included in the dependency lock. The prepared six-page report
was rendered and visually verified.

## Validate existing outputs without fitting

```text
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer validates both CSV contracts and generates the fixed December chart.
It does not measure accuracy. All original inputs remain in the repository
root; the commands use their supplied filenames and do not overwrite them.
