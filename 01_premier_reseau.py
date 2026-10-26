"""
Premier réseau PyTorch : reconnaître des chiffres manuscrits (images 8x8 de scikit-learn).
Le squelette (données -> modèle -> perte -> boucle d'entraînement -> évaluation) est
EXACTEMENT celui qu'on réutilisera pour les signaux ultrasonores.
"""
import numpy as np
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
chiffres = load_digits()
X = (chiffres.data / 16.0).astype(np.float32)        # pixels entre 0 et 1
y = chiffres.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte = torch.from_numpy(Xtr), torch.from_numpy(Xte)
ytr, yte = torch.from_numpy(ytr), torch.from_numpy(yte)

modele = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
opt = torch.optim.Adam(modele.parameters(), lr=1e-2)
perte = nn.CrossEntropyLoss()

for epoch in range(1, 31):
    opt.zero_grad()
    l = perte(modele(Xtr), ytr)
    l.backward()
    opt.step()
    if epoch % 5 == 0:
        with torch.no_grad():
            acc = (modele(Xte).argmax(1) == yte).float().mean().item()
        print(f"epoch {epoch:2d}  perte {l.item():.3f}  exactitude test {acc:.3f}")
