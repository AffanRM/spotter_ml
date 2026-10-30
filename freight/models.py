"""Small, explicit candidate set with fixed parameters and training-only fits."""

from dataclasses import asdict, dataclass

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freight.data import TARGET
from freight.features import CATEGORICAL, build_features


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    feature_set: str = "core"
    loss: str = "MAE"


MODEL_SPECS = [
    ModelSpec("median_rate", "median"),
    ModelSpec("equipment_rate_per_mile", "equipment_rpm"),
    ModelSpec("ridge_core_rate_per_mile", "ridge"),
    ModelSpec("catboost_core_mae", "catboost"),
    ModelSpec("catboost_market_mae", "catboost", "market"),
    ModelSpec("catboost_full_mae", "catboost", "full"),
    ModelSpec("catboost_core_rmse", "catboost", "core", "RMSE"),
    ModelSpec("catboost_chart_mae", "catboost", "chart"),
]


class RateModel:
    def __init__(self, spec: ModelSpec, *, iterations=500, seed=42, threads=4):
        self.spec = spec
        self.iterations = iterations
        self.seed = seed
        self.threads = threads

    def fit(self, frame: pd.DataFrame):
        y = frame[TARGET].to_numpy(dtype=float)
        if len(y) == 0 or not np.isfinite(y).all() or (y <= 0).any():
            raise ValueError("Training targets must be finite and positive")
        if self.spec.kind == "median":
            self.median_ = float(np.median(y))
        elif self.spec.kind == "equipment_rpm":
            rpm = frame[TARGET] / frame.distance
            self.global_rpm_ = float(rpm.median())
            self.equipment_rpm_ = rpm.groupby(frame.equipment).median().to_dict()
        elif self.spec.kind == "ridge":
            x = build_features(frame, self.spec.feature_set)
            numeric = [c for c in x.columns if c not in CATEGORICAL]
            preprocessing = ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ])
            self.estimator_ = make_pipeline(preprocessing, Ridge(alpha=10, solver="lsqr"))
            self.estimator_.fit(x, y / frame.distance.to_numpy())
        elif self.spec.kind == "catboost":
            x = build_features(frame, self.spec.feature_set)
            self.estimator_ = CatBoostRegressor(
                iterations=self.iterations, depth=6, learning_rate=.05,
                loss_function=self.spec.loss, random_seed=self.seed,
                thread_count=self.threads, verbose=False, allow_writing_files=False,
                l2_leaf_reg=5,
            )
            # Fixed iteration count: no validation labels used for early stopping.
            self.estimator_.fit(x, y, cat_features=CATEGORICAL)
        else:
            raise ValueError(f"Unknown model kind: {self.spec.kind}")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.kind == "median":
            prediction = np.full(len(frame), self.median_)
        elif self.spec.kind == "equipment_rpm":
            prediction = frame.equipment.map(self.equipment_rpm_).fillna(self.global_rpm_).to_numpy() * frame.distance.to_numpy()
        else:
            prediction = np.asarray(self.estimator_.predict(build_features(frame, self.spec.feature_set)))
            if self.spec.kind == "ridge":
                prediction = prediction * frame.distance.to_numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Model produced nonfinite predictions")
        return np.maximum(prediction, .01)

    def configuration(self):
        return {**asdict(self.spec), "iterations": self.iterations, "seed": self.seed, "threads": self.threads}
