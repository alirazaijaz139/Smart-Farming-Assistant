import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = Path(r"C:\Users\User\Downloads\archive (10)\plantvillage dataset\color")
DST = BASE_DIR / "data" / "plant_disease"

CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]
PER_CLASS = 300
random.seed(42)

for cls in CLASSES:
    images = sorted((SRC / cls).glob("*.*"))
    random.shuffle(images)
    images = images[:PER_CLASS]
    split = int(len(images) * 0.8)
    for name, subset in (("train", images[:split]), ("val", images[split:])):
        out = DST / name / cls
        out.mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, out / img.name)
    print(cls, len(images))