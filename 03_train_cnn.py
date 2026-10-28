"""Entraîne le CNN 1D et sauvegarde le meilleur modèle (selon la validation)."""
import sys, time
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulateur import jeux_standard
from modele import CNN1D

# Choix du device : la puce graphique Apple (MPS) si elle est disponible, sinon le processeur.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device :", device)

torch.manual_seed(0)
np.random.seed(0)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LR, BATCH = 1e-3, 64

# Augmentation de données (étape 6) : à chaque mini-lot, on perturbe les signaux d'entraînement
# avec un gain et un bruit tirés au hasard, pour que le modèle apprenne à les ignorer.
AUGMENTER = True
GAIN_MIN, GAIN_MAX = 0.7, 1.3      # erreur d'étalonnage simulée : ±30 % sur l'amplitude
BRUIT_MAX = 0.1                    # écart-type du bruit ajouté, tiré entre 0 et BRUIT_MAX

# Entraînement adverse (étape 6, exercice 3) : on ajoute à chaque mini-lot sa version attaquée
# par FGSM, calculée avec le modèle du moment, pour que le réseau apprenne à y résister.
ADVERSAIRE = False                 # essai fait à l'étape 6 : voir reponses_exercices.md
EPS_ADV = 0.02                     # taille de l'attaque utilisée pendant l'entraînement

(Xtr, ytr), (Xva, yva), _ = jeux_standard()
tens = lambda X, y: (torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y).float())
Xtr_t, ytr_t = tens(Xtr, ytr)
Xva_t, yva_t = tens(Xva, yva)
# Données envoyées une seule fois sur le device, avant la boucle.
Xtr_t, ytr_t = Xtr_t.to(device), ytr_t.to(device)
Xva_t, yva_t = Xva_t.to(device), yva_t.to(device)

modele = CNN1D().to(device)                         # le modèle doit être sur le même device que les données
print("Nombre de paramètres :", sum(p.numel() for p in modele.parameters()))
opt = torch.optim.Adam(modele.parameters(), lr=LR)
perte = nn.BCEWithLogitsLoss()

hist = {"train": [], "val": [], "acc_val": []}
meilleur = float("inf")
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    modele.train()
    ordre = torch.randperm(len(Xtr_t), device=device)
    tot = 0.0
    for i in range(0, len(ordre), BATCH):
        idx = ordre[i:i + BATCH]
        opt.zero_grad()
        xb = Xtr_t[idx]
        if AUGMENTER:
            n = len(idx)
            gain = torch.empty(n, 1, 1, device=device).uniform_(GAIN_MIN, GAIN_MAX)   # un gain par signal
            ecart = torch.empty(n, 1, 1, device=device).uniform_(0, BRUIT_MAX)       # un niveau de bruit par signal
            xb = xb * gain + ecart * torch.randn_like(xb)
        yb = ytr_t[idx]
        if ADVERSAIRE:
            xa = xb.detach().clone().requires_grad_(True)
            perte(modele(xa), yb).backward()            # gradient de la perte par rapport à l'ENTRÉE
            xa = (xa + EPS_ADV * xa.grad.sign()).detach() # version attaquée de chaque signal
            opt.zero_grad()                              # on jette les gradients des poids calculés pour l'attaque
            xb = torch.cat([xb, xa])                     # le lot contient les signaux normaux ET attaqués
            yb = torch.cat([yb, yb])                     # avec les mêmes étiquettes
        l = perte(modele(xb), yb)
        l.backward()
        opt.step()
        tot += l.item() * len(idx)
    modele.eval()
    with torch.no_grad():
        lo = modele(Xva_t)
        lv = perte(lo, yva_t).item()
        acc = ((lo > 0).float() == yva_t).float().mean().item()
    hist["train"].append(tot / len(Xtr_t)); hist["val"].append(lv); hist["acc_val"].append(acc)
    marque = ""
    if lv < meilleur:
        meilleur = lv
        torch.save({k: v.cpu() for k, v in modele.state_dict().items()}, "sorties/cnn_meilleur.pt")  # poids ramenés sur CPU
        marque = "  <- sauvegardé"
    print(f"epoch {ep:2d}  perte train {hist['train'][-1]:.4f}  perte val {lv:.4f}  acc val {acc:.3f}{marque}")
if device.type == "mps":
    torch.mps.synchronize()                         # le GPU travaille en décalé : on attend qu'il ait fini
print(f"Durée : {time.time() - t0:.0f} s  (device : {device})")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["train"], label="entraînement"); ax[0].plot(hist["val"], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("perte"); ax[0].legend()
ax[1].plot(hist["acc_val"]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("exactitude (validation)")
plt.tight_layout(); plt.savefig("sorties/courbes_entrainement.png", dpi=110)
