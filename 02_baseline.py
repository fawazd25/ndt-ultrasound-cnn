"""Deux méthodes SANS réseau de neurones, pour avoir un point de comparaison."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from simulateur import jeux_standard

(Xtr, ytr), (Xva, yva), (Xte, yte) = jeux_standard()

# --- Baseline 1 : la "porte" (gate) fixe, comme en inspection classique ---------
# Un technicien règle une fenêtre de temps et un seuil : alarme si un écho
# dépasse le seuil dans la fenêtre. Ici la fenêtre couvre 100..230.
G0, G1 = 100, 230
score_porte = lambda X: np.abs(X[:, G0:G1]).max(axis=1)
seuils = np.linspace(0.02, 0.6, 120)
meilleur = max(seuils, key=lambda s: ((score_porte(Xva) > s) == yva).mean())
pred = score_porte(Xte) > meilleur
print(f"[porte fixe]  seuil={meilleur:.3f}  exactitude={(pred == yte).mean():.3f}  "
      f"rappel={recall_score(yte, pred):.3f}  précision={precision_score(yte, pred):.3f}  "
      f"AUC={roc_auc_score(yte, score_porte(Xte)):.3f}")

# --- Baseline 2 : régression logistique sur le signal brut ----------------------
lr = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
p = lr.predict_proba(Xte)[:, 1]
pred = p > 0.5
print(f"[logistique]  exactitude={(pred == yte).mean():.3f}  "
      f"rappel={recall_score(yte, pred):.3f}  précision={precision_score(yte, pred):.3f}  "
      f"AUC={roc_auc_score(yte, p):.3f}")
