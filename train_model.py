import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BRANDS = ["Toyota", "Hyundai", "Suzuki", "BMW", "Mercedes", "Volkswagen"]
FUEL_TYPES = ["petrol", "diesel", "hybrid", "ev"]


def generate_synthetic_car_data(n_samples: int = 5000, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    mileage = rng.uniform(5_000, 320_000, size=n_samples)
    age = rng.uniform(0.5, 20, size=n_samples)  # years
    engine_size = rng.uniform(1.0, 4.5, size=n_samples)  # liters
    horsepower = rng.uniform(60, 380, size=n_samples)
    doors = rng.integers(3, 6, size=n_samples)  # 3–5 doors

    brand_idx = rng.integers(0, len(BRANDS), size=n_samples)
    fuel_idx = rng.integers(0, len(FUEL_TYPES), size=n_samples)

    brand_arr = np.array(BRANDS, dtype=object)[brand_idx]
    fuel_arr = np.array(FUEL_TYPES, dtype=object)[fuel_idx]

    base_price = 18_000
    brand_factor = {
        "Toyota": 1.0,
        "Hyundai": 0.9,
        "Suzuki": 0.85,
        "BMW": 1.8,
        "Mercedes": 2.0,
        "Volkswagen": 1.1,
    }
    fuel_factor = {
        "petrol": 1.0,
        "diesel": 1.05,
        "hybrid": 1.2,
        "ev": 0.75,  # lower effective price due to lower taxes/incentives
    }

    price = (
        base_price
        * np.vectorize(brand_factor.get)(brand_arr)
        * np.vectorize(fuel_factor.get)(fuel_arr)
        - 0.03 * mileage
        - 800 * age
        + 4_000 * (engine_size - 1.6)
        + 25 * horsepower
        + 300 * (doors - 4)
    )

    noise = rng.normal(0, 3_000, size=n_samples)
    y = np.clip(price + noise, 2_000, 120_000)

    num_features = np.column_stack([mileage, age, engine_size, horsepower, doors])

    brand_one_hot = np.zeros((n_samples, len(BRANDS)))
    brand_one_hot[np.arange(n_samples), brand_idx] = 1
    fuel_one_hot = np.zeros((n_samples, len(FUEL_TYPES)))
    fuel_one_hot[np.arange(n_samples), fuel_idx] = 1

    X = np.hstack([num_features, brand_one_hot, fuel_one_hot])
    feature_names = [
        "mileage",
        "age",
        "engine_size",
        "horsepower",
        "doors",
        *[f"brand_{b.lower()}" for b in BRANDS],
        *[f"fuel_{f}" for f in FUEL_TYPES],
    ]

    return X, y, feature_names, BRANDS, FUEL_TYPES


def train_and_save_model():
    X, y, feature_names, brands, fuel_types = generate_synthetic_car_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_log = np.log(y_train)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse**0.5)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "car_price_model.pkl"
    metadata_path = models_dir / "metadata.json"

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "brand_categories": brands,
            "fuel_categories": fuel_types,
            "target_transform": "log",
        },
        model_path,
    )

    metadata = {
        "task": "car_price_prediction",
        "target": "price_usd",
        "feature_names": feature_names,
        "brand_categories": brands,
        "fuel_categories": fuel_types,
        "target_transform": "log",
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "model_type": "RandomForestRegressor",
        "n_estimators": model.n_estimators,
        "random_state": model.random_state,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model trained and saved.")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Metadata path: {os.path.abspath(metadata_path)}")


if __name__ == "__main__":
    train_and_save_model()

