"""Step 3: severity classifier (ResNet18). Folder layout:
data/severity/train/{minor,moderate,severe}/*.jpg and data/severity/val/... """
import torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tr = T.Compose([T.Resize((224,224)),T.RandomHorizontalFlip(),T.ColorJitter(.2,.2,.2),T.ToTensor(),norm])
va = T.Compose([T.Resize((224,224)),T.ToTensor(),norm])
dtr = datasets.ImageFolder("data/severity/train", tr); dva = datasets.ImageFolder("data/severity/val", va)
print("classes:", dtr.classes)  # must be ['minor','moderate','severe']
ltr = DataLoader(dtr, 32, shuffle=True, num_workers=2); lva = DataLoader(dva, 32)
m = models.resnet18(weights="IMAGENET1K_V1"); m.fc = nn.Linear(512, 3); m.to(dev)
opt = torch.optim.AdamW(m.parameters(), 3e-4); lossf = nn.CrossEntropyLoss(); best = 0
for ep in range(15):
    m.train()
    for x, y in ltr:
        x, y = x.to(dev), y.to(dev); opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    m.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in lva:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
    acc = ok / n; print(f"epoch {ep} val_acc {acc:.3f}")
    if acc > best: best = acc; torch.save(m.state_dict(), "models/severity.pt")
