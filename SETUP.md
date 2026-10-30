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
```

The benchmark defaults to 500 CatBoost iterations and four CPU threads. Override
with `python -m freight.benchmark --iterations 500 --threads 4` if desired.
Changing the iteration count produces a new experiment, so retain that setting
when comparing results. It does not evaluate the reserved final-check window.

## Recreate on another computer

With Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements.txt` is the unchanged employer-supplied scorer dependency list.
`requirements-project.txt` includes it and adds scientific computing,
scikit-learn baselines, CatBoost, model persistence, and PDF report tools.
CatBoost is a candidate model to evaluate, not a model selection decision.
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
The scorer writes `scorer_results/candidate_december.png`. This command is for
later: prediction files have not been generated during environment setup.
