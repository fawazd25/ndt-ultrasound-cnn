"""Le modèle tient-il quand les conditions changent ? (pont avec la cybersécurité)"""
import os, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulateur import jeux_standard, ZONE_MORTE
from modele import CNN1D

_, _, (Xte, yte) = jeux_standard()
modele = CNN1D()
# Modèle à tester : sorties/cnn_meilleur.pt par défaut, ou un autre fichier passé en argument,
# par exemple : python 05_robustesse.py sorties/cnn_sans_augmentation.pt
CHEMIN = sys.argv[1] if len(sys.argv) > 1 else "sorties/cnn_meilleur.pt"
modele.load_state_dict(torch.load(CHEMIN))
print(f"Modèle testé : {CHEMIN}")
modele.eval()
rng = np.random.default_rng(0)


def evaluer(X):
    with torch.no_grad():
        pred = (modele(torch.from_numpy(X.astype(np.float32)).unsqueeze(1)) > 0).numpy()
    exactitude = (pred == yte).mean()
    rappel = pred[yte == 1].mean()
    return exactitude, rappel


print(f"{'perturbation':<32}{'exactitude':>11}{'rappel':>9}")
ref = evaluer(Xte)
print(f"{'aucune':<32}{ref[0]:>11.3f}{ref[1]:>9.3f}")

# 1) Bruit supplémentaire (électronique vieillissante, câble abîmé...)
res_bruit = []
for s in [0.02, 0.05, 0.1, 0.2]:
    r = evaluer(Xte + rng.normal(0, s, Xte.shape))
    res_bruit.append(r[0]); print(f"{'bruit +' + str(s):<32}{r[0]:>11.3f}{r[1]:>9.3f}")

# 2) Mauvais étalonnage : amplitude globale trop faible / trop forte
for f in [0.7, 1.3]:
    r = evaluer(Xte * f); print(f"{'amplitude x' + str(f):<32}{r[0]:>11.3f}{r[1]:>9.3f}")

# 3) Décalage temporel (sonde posée un peu autrement)
for d in [-15, 15]:
    Xd = np.roll(Xte, d, axis=1); Xd[:, :ZONE_MORTE] = 0
    r = evaluer(Xd); print(f"{'décalage ' + str(d) + ' échantillons':<32}{r[0]:>11.3f}{r[1]:>9.3f}")

# 4) Attaque adverse FGSM : on déplace chaque signal d'un tout petit pas eps,
#    dans la direction qui augmente le plus l'erreur du modèle.
perte = nn.BCEWithLogitsLoss()
x = torch.from_numpy(Xte).unsqueeze(1).clone().requires_grad_(True)
y = torch.from_numpy(yte).float()
perte(modele(x), y).backward()
direction = x.grad.sign().detach()
res_fgsm = []
for eps in [0.01, 0.02, 0.05, 0.1]:
    xa = (x.detach() + eps * direction).squeeze(1).numpy()
    r = evaluer(xa); res_fgsm.append(r[0])
    print(f"{'attaque adverse eps=' + str(eps):<32}{r[0]:>11.3f}{r[1]:>9.3f}")

# 5) Même attaque, mais la zone morte reste à zéro (exercice 2 de l'étape 6).
#    Dans une vraie chaîne, la normalisation remet ces 100 premiers échantillons à zéro :
#    une perturbation placée là serait effacée avant d'atteindre le modèle.
direction_zm = direction.clone()
direction_zm[:, :, :ZONE_MORTE] = 0
res_fgsm_zm = []
for eps in [0.01, 0.02, 0.05, 0.1]:
    xa = (x.detach() + eps * direction_zm).squeeze(1).numpy()
    r = evaluer(xa); res_fgsm_zm.append(r[0])
    print(f"{'attaque zone morte à 0 eps=' + str(eps):<32}{r[0]:>11.3f}{r[1]:>9.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot([0.02, 0.05, 0.1, 0.2], res_bruit, "o-", label="bruit aléatoire")
ax.plot([0.01, 0.02, 0.05, 0.1], res_fgsm, "s-", label="attaque adverse (FGSM)")
ax.plot([0.01, 0.02, 0.05, 0.1], res_fgsm_zm, "^--", label="FGSM, zone morte à 0")
ax.axhline(ref[0], ls="--", c="gray"); ax.set_xlabel("amplitude de la perturbation")
ax.set_ylabel("exactitude"); ax.legend(); plt.tight_layout()
nom = os.path.splitext(os.path.basename(CHEMIN))[0]
image = "sorties/robustesse.png" if nom == "cnn_meilleur" else f"sorties/robustesse_{nom}.png"
plt.savefig(image, dpi=110)
print(f"Image écrite : {image}")
