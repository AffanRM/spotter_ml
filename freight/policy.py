"""Load the model choices recorded before the final-check evaluation."""

import json
from pathlib import Path

from freight.models import MODEL_SPECS, RateModel


ROOT = Path(__file__).resolve().parents[1]


def load_policy(path: Path = ROOT / "model_policy.json") -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("policy_version") != 1:
        raise ValueError("Unsupported model policy version")
    specs = {spec.name: spec for spec in MODEL_SPECS}
    for key in ["main_model", "chart_model", "reference_model"]:
        if policy.get(key) not in specs:
            raise ValueError(f"Unknown {key} in model policy")
    main, chart = specs[policy["main_model"]], specs[policy["chart_model"]]
    if main.kind != "catboost" or main.feature_set != "full":
        raise ValueError("Main model must use the declared full-feature path")
    if chart.kind != "catboost" or chart.feature_set != "chart":
        raise ValueError("Chart model must use the reduced-input path")
    if specs[policy["reference_model"]].kind != "equipment_rpm":
        raise ValueError("Expected the predeclared equipment baseline")
    for key in ["iterations", "threads"]:
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError(f"Policy {key} must be a positive integer")
    if type(policy.get("seed")) is not int:
        raise ValueError("Policy seed must be an integer")
    if policy.get("unseen_city_routing") is not False or policy.get("drop_target_outliers") is not False:
        raise ValueError("This policy does not implement city routing or target deletion")
    if (policy.get("train_end"), policy.get("validation_end")) != ("2025-08-31", "2025-10-31"):
        raise ValueError("Unexpected final-check date boundaries")
    return policy


def make_policy_model(policy: dict, role: str) -> RateModel:
    if role not in ["main", "chart", "reference"]:
        raise ValueError(f"Unknown model role: {role}")
    name = policy[f"{role}_model"]
    spec = next(spec for spec in MODEL_SPECS if spec.name == name)
    return RateModel(spec, iterations=policy["iterations"], seed=policy["seed"], threads=policy["threads"])
