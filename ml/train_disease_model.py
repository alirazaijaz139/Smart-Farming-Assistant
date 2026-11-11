import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data" / "plant_disease"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(DATA / "train", train_tf)
val_ds = datasets.ImageFolder(DATA / "val", val_tf)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for p in model.features.parameters():
    p.requires_grad = False                      # freeze the pretrained backbone
model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))
model.to(device)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | "
          f"loss {total_loss / len(train_ds):.4f} | "
          f"val acc {correct / len(val_ds):.4f}")

torch.save(model.state_dict(), BASE_DIR / "ml" / "disease_model.pt")
(BASE_DIR / "ml" / "disease_classes.json").write_text(json.dumps(train_ds.classes))
print("Saved model and class names")