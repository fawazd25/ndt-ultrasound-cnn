"""Évalue le CNN sur le jeu de TEST (jamais vu pendant l'entraînement)."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from simulateur import jeux_standard
from modele import CNN1D

_, (Xva, yva), (Xte, yte) = jeux_standard()
modele = CNN1D()
modele.load_state_dict(torch.load("sorties/cnn_meilleur.pt"))
modele.eval()


def probas(X):
    with torch.no_grad():
        return torch.sigmoid(modele(torch.from_numpy(X).unsqueeze(1))).numpy()


def rapport(y, p, seuil, titre):
    pred = p >= seuil
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rappel = tp / (tp + fn)            # parmi les VRAIS défauts, combien détectés
    precision = tp / (tp + fp)         # parmi les alarmes, combien sont de vrais défauts
    fausses_alarmes = fp / (fp + tn)   # parmi les pièces saines, combien d'alarmes
    print(f"{titre:<34} seuil={seuil:.2f}  rappel={rappel:.3f}  précision={precision:.3f}  "
          f"fausses alarmes={fausses_alarmes:.3f}  (défauts ratés : {fn}/{fn + tp})")
    return confusion_matrix(y, pred)


pva, pte = probas(Xva), probas(Xte)
print(f"AUC (test) = {roc_auc_score(yte, pte):.3f}\n")
cm = rapport(yte, pte, 0.5, "Seuil standard 0.5")

# --- Intervalle de confiance sur l'exactitude (bootstrap) ---
correct = (pte >= 0.5) == yte               # True si la prédiction est juste, pour chacun des 2000 signaux
rng = np.random.default_rng(0)              # graine fixe : résultat reproductible
exactitudes = []
for _ in range(1000):                       # 1000 tirages
    idx = rng.integers(0, len(yte), len(yte))   # 2000 indices tirés AVEC remise
    exactitudes.append(correct[idx].mean())     # exactitude de ce tirage
bas, haut = np.percentile(exactitudes, [2.5, 97.5])
print(f"Exactitude (test, seuil 0.5) = {correct.mean():.3f}  "
      f"intervalle de confiance à 95 % : [{bas:.3f} ; {haut:.3f}]\n")

# Choisir le seuil sur la VALIDATION (pas sur le test, sinon on triche) :
# le plus haut seuil qui garantit un rappel >= 95 %.
CIBLE = 0.95
candidats = np.linspace(0.01, 0.99, 197)
ok = [s for s in candidats if ((pva >= s) & (yva == 1)).sum() / (yva == 1).sum() >= CIBLE]
seuil_rappel = round(float(max(ok)), 2)
rapport(yte, pte, seuil_rappel, f"Seuil choisi pour rappel >= {CIBLE:.0%}")
json.dump({"seuil": seuil_rappel, "cible_rappel": CIBLE}, open("sorties/seuil.json", "w"))
print(f"\nSeuil sauvegardé dans sorties/seuil.json : {seuil_rappel:.2f}")

fpr, tpr, _ = roc_curve(yte, pte)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
ax[0].set_xlabel("fausses alarmes"); ax[0].set_ylabel("rappel"); ax[0].set_title("Courbe ROC")
ax[1].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax[1].text(j, i, v, ha="center", va="center", fontsize=14)
ax[1].set_xticks([0, 1], ["saine", "défaut"]); ax[1].set_yticks([0, 1], ["saine", "défaut"])
ax[1].set_xlabel("prédit"); ax[1].set_ylabel("réel"); ax[1].set_title("Matrice de confusion (seuil 0.5)")
plt.tight_layout(); plt.savefig("sorties/evaluation.png", dpi=110)
