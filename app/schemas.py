from typing import Dict, Literal

from pydantic import BaseModel, Field, conint, confloat


class RegressionFeatures(BaseModel):
    mileage: confloat(ge=0, le=320_000) = Field(
        ...,
        description="Total mileage on the car in kilometers",
        example=60_000,
    )
    age: confloat(ge=0, le=30) = Field(
        ...,
        description="Age of the car in years",
        example=5,
    )
    engine_size: confloat(ge=0.8, le=6.0) = Field(
        ...,
        description="Engine size in liters",
        example=2.0,
    )
    horsepower: confloat(ge=40, le=600) = Field(
        ...,
        description="Engine power in HP",
        example=150,
    )
    doors: conint(ge=2, le=5) = Field(
        ...,
        description="Number of doors",
        example=4,
    )

    brand: Literal["Toyota", "Hyundai", "Suzuki", "BMW", "Mercedes", "Volkswagen"] = (
        Field(
            ...,
            description="Car brand",
            example="Toyota",
        )
    )
    fuel_type: Literal["petrol", "diesel", "hybrid", "ev"] = Field(
        ...,
        description="Fuel type of the car",
        example="petrol",
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(
        ..., description="Predicted car price in USDT (treated as USD value)"
    )
    currency: str = Field(default="USDT", description="Base currency of the prediction")
    etb_price: float = Field(
        ..., description="Predicted car price converted to Ethiopian Birr"
    )
    etb_currency: str = Field(default="ETB", description="Secondary currency (Birr)")
    usdt_to_etb_rate: float = Field(
        ..., description="Conversion rate used: 1 USDT = this many ETB"
    )
    feature_importance: Dict[str, float] = Field(
        ..., description="Relative importance of each feature (sums to ~1.0)"
    )
    model_metadata: Dict[str, object] = Field(
        ..., description="Information about the underlying regression model"
    )

