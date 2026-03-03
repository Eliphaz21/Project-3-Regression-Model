from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model import get_default_model
from .schemas import PredictionResponse, RegressionFeatures


USDT_TO_ETB_RATE = 155.95


app = FastAPI(
    title="Car Price Regression API",
    description="Predict car prices using a trained regression model.",
    version="1.0.0",
)


BASE_DIR = Path(__file__).resolve().parent.parent
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


model = get_default_model()


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/regression", response_model=PredictionResponse)
def predict_regression(features: RegressionFeatures) -> PredictionResponse:
    try:
        feature_values = features.dict()
        predicted_price, importance = model.predict(feature_values)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    usdt_price = predicted_price
    etb_price = usdt_price * USDT_TO_ETB_RATE

    return PredictionResponse(
        predicted_price=usdt_price,
        currency="USDT",
        etb_price=etb_price,
        etb_currency="ETB",
        usdt_to_etb_rate=USDT_TO_ETB_RATE,
        feature_importance=importance,
        model_metadata=model.metadata,
    )

