import io
import json
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .. import models as db_models
from ..security import get_current_user

router = APIRouter(prefix="/disease", tags=["Disease Detection"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLASSES = json.loads((BASE_DIR / "ml" / "disease_classes.json").read_text())

net = models.mobilenet_v2(weights=None)
net.classifier[1] = nn.Linear(net.last_channel, len(CLASSES))
net.load_state_dict(torch.load(BASE_DIR / "ml" / "disease_model.pt", map_location="cpu"))
net.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@router.post("/detect")
async def detect_disease(
    file: UploadFile = File(...),
    user: db_models.User = Depends(get_current_user),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPG or PNG image")

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(net(tf(img).unsqueeze(0)), dim=1)[0]

    top = torch.topk(probs, 3)
    results = [
        {"label": CLASSES[i], "confidence": round(float(p), 3)}
        for p, i in zip(top.values, top.indices)
    ]
    return {
        "prediction": results[0]["label"],
        "confident": results[0]["confidence"] >= 0.6,
        "top_3": results,
    }