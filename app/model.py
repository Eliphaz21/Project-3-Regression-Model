import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np


class RegressionModel:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self._model = None
        self._feature_names: List[str] = []
        self._metadata: Dict[str, object] = {}
        self._brand_categories: List[str] = []
        self._fuel_categories: List[str] = []
        self._target_transform: str | None = None

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Run 'python train_model.py' first to train and save the model."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.metadata_path}. "
                f"Run 'python train_model.py' first to generate metadata."
            )

        bundle = joblib.load(self.model_path)
        self._model = bundle["model"]
        self._feature_names = bundle["feature_names"]
        self._brand_categories = bundle.get("brand_categories", [])
        self._fuel_categories = bundle.get("fuel_categories", [])
        self._target_transform = bundle.get("target_transform")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def metadata(self) -> Dict[str, object]:
        return self._metadata

    def predict(self, feature_values: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
        mileage = float(feature_values["mileage"])
        age = float(feature_values["age"])
        engine_size = float(feature_values["engine_size"])
        horsepower = float(feature_values["horsepower"])
        doors = float(feature_values["doors"])
        brand = str(feature_values.get("brand", "")).strip()
        fuel_type = str(feature_values.get("fuel_type", "")).strip()

        num_features = [mileage, age, engine_size, horsepower, doors]

        brand_one_hot = [
            1.0 if brand == b else 0.0 for b in self._brand_categories
        ]
        fuel_one_hot = [
            1.0 if fuel_type == f else 0.0 for f in self._fuel_categories
        ]

        ordered_values = num_features + brand_one_hot + fuel_one_hot
        X = np.array(ordered_values, dtype=float).reshape(1, -1)

        raw_pred = float(self._model.predict(X)[0])
        if self._target_transform == "log":
            prediction = float(np.exp(raw_pred))
        else:
            prediction = raw_pred

        importance: Dict[str, float] = {}
        if hasattr(self._model, "feature_importances_"):
            raw_importances = np.array(self._model.feature_importances_, dtype=float)
            total = raw_importances.sum()
            if total > 0:
                normalized = raw_importances / total
            else:
                normalized = raw_importances
            importance = {
                name: float(score) for name, score in zip(self._feature_names, normalized)
            }

        return prediction, importance


def get_default_model() -> RegressionModel:
    models_dir = Path("models")
    model_path = models_dir / "car_price_model.pkl"
    metadata_path = models_dir / "metadata.json"
    return RegressionModel(model_path=model_path, metadata_path=metadata_path)

