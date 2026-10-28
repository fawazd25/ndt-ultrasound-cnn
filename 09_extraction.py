"""
Démonstration d'extraction de modèle (étape 10, exercice 5).

On se met à la place d'un attaquant qui n'a QUE l'accès à l'API /diagnostic :
il ne connaît ni les poids, ni l'architecture, ni les étiquettes de ses signaux.
  1. Il interroge l'API avec ses propres signaux et note les réponses.
  2. Il entraîne sa propre copie du modèle sur ces réponses.
  3. Il attaque sa copie (FGSM, en boîte blanche) et envoie les signaux trompeurs à l'API.

On compare ce que l'attaquant obtient avec la probabilité exacte, et avec la décision seule.
Aucun vrai serveur n'est lancé : les requêtes passent par le client de test de Flask.
Le modèle de référence (sorties/cnn_meilleur.pt) n'est ni modifié ni lu directement.
"""
import importlib.util
import time
import numpy as np
import torch
import torch.nn as nn
from simulateur import generer, normaliser, ZONE_MORTE
from modele import CNN1D

torch.manual_seed(0)
rng = np.random.default_rng(0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# L'API, vue exactement comme un client la voit.
spec = importlib.util.spec_from_file_location("api_flask", "08_api_flask.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
client = api.app.test_client()
compteur = {"requetes": 0}


def interroger(X_brut):
    """Envoie chaque signal brut à l'API ; renvoie les probabilités reçues et le seuil."""
    probas, seuil = np.empty(len(X_brut)), None
    for i, x in enumerate(X_brut):
        r = client.post("/diagnostic", json={"signal": x.tolist()}).get_json()
        probas[i], seuil = r["probabilite_defaut"], r["seuil"]
    compteur["requetes"] += len(X_brut)
    return probas, seuil


def entrainer_copie(Xn, cibles, epochs=30):
    """L'attaquant choisit sa propre architecture : celle de départ du projet (4 blocs, noyau 9)."""
    torch.manual_seed(0)
    m = CNN1D(canaux=(8, 16, 32, 32), noyau=9).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    perte = nn.BCEWithLogitsLoss()       # accepte des cibles « douces » (probabilités) ou 0/1
    X = torch.from_numpy(Xn).unsqueeze(1).to(device)
    y = torch.from_numpy(cibles.astype(np.float32)).to(device)
    for _ in range(epochs):
        ordre = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 64):
            idx = ordre[i:i + 64]
            opt.zero_grad()
            perte(m(X[idx]), y[idx]).backward()
            opt.step()
    return m.to("cpu").eval()


def probas_copie(m, Xn):
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(Xn).unsqueeze(1))).numpy()


t0 = time.time()

# --- 1) Les signaux de l'attaquant : bruts, SANS étiquettes (graine différente des jeux du projet)
X_att, _ = generer(6000, seed=50)
p_att, SEUIL = interroger(X_att)
print(f"Seuil révélé par l'API : {SEUIL}")

# Signaux de test (les mêmes que le jeu de test du projet), pour mesurer la copie
X_te, y_te = generer(2000, seed=3)
p_vic, _ = interroger(X_te)
d_vic = p_vic >= SEUIL
Xn_att, Xn_te = normaliser(X_att), normaliser(X_te)
print(f"Modèle original (via l'API) : exactitude {((p_vic >= 0.5) == y_te).mean():.3f}\n")

# --- 2) Entraîner une copie avec la probabilité exacte, puis avec la décision seule
print(f"{'requêtes':>9}  {'réponses utilisées':<20}{'accord avec l API':>18}{'exactitude':>12}")
copies = {}
for n in [500, 2000, 6000]:
    for nom, cibles in [("probabilité", p_att[:n]), ("décision seule", (p_att[:n] >= SEUIL).astype(float))]:
        m = entrainer_copie(Xn_att[:n], cibles)
        pc = probas_copie(m, Xn_te)
        accord = ((pc >= SEUIL) == d_vic).mean()
        exact = ((pc >= 0.5) == y_te).mean()
        print(f"{n:>9}  {nom:<20}{accord:>18.3f}{exact:>12.3f}")
        copies[(n, nom)] = m

# --- 3) Attaque par transfert : FGSM sur la copie, signaux envoyés à l'API
copie = copies[(6000, "probabilité")]
cibles = np.where((y_te == 1) & d_vic)[0]          # vrais défauts que l'API détecte
xn = torch.from_numpy(Xn_te[cibles]).unsqueeze(1).requires_grad_(True)
nn.BCEWithLogitsLoss()(copie(xn), torch.ones(len(cibles))).backward()
direction = xn.grad.sign().squeeze(1).numpy()      # pousse chaque signal vers « saine »
direction[:, :ZONE_MORTE] = 0                      # la perturbation doit survivre à la normalisation
hasard = rng.choice([-1.0, 1.0], size=direction.shape)
hasard[:, :ZONE_MORTE] = 0
ref = np.abs(X_te[cibles, :90]).max(axis=1, keepdims=True)   # échelle utilisée par normaliser()

print(f"\nAttaque par transfert sur {len(cibles)} défauts que l'API détecte :")
print(f"{'eps':>6}{'défauts devenus « saine » (attaque)':>38}{'(bruit au hasard)':>20}")
for eps in [0.02, 0.05]:
    p_adv, _ = interroger(X_te[cibles] + eps * ref * direction)
    p_bru, _ = interroger(X_te[cibles] + eps * ref * hasard)
    print(f"{eps:>6}{(p_adv < SEUIL).mean():>38.1%}{(p_bru < SEUIL).mean():>20.1%}")

print(f"\nRequêtes envoyées à l'API : {compteur['requetes']} en {time.time() - t0:.0f} s, "
      f"toutes inscrites dans sorties/requetes.log")
