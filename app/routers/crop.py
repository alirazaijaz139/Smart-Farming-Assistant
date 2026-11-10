from pathlib import Path
import joblib
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import models
from ..security import get_current_user

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model = joblib.load(BASE_DIR / "ml" / "crop_model.joblib")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class CropInput(BaseModel):
    N: float = Field(ge=0, le=200)
    P: float = Field(ge=0, le=200)
    K: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=500)


@router.post("/recommend")
def recommend_crop(
    data: CropInput,
    user: models.User = Depends(get_current_user),
):
    row = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)
    probs = model.predict_proba(row)[0]
    top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return {
        "recommended_crop": top3[0][0],
        "top_3": [{"crop": c, "confidence": round(float(p), 3)} for c, p in top3],
    }