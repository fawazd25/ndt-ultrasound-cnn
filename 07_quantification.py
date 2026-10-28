"""Optimisation : élagage (pruning) et quantification INT8. On mesure précision, taille, vitesse."""
import os, time, copy, json
import numpy as np
import torch
import torch.nn.utils.prune as prune
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat)
from simulateur import jeux_standard
from modele import CNN1D

(Xtr, ytr), _, (Xte, yte) = jeux_standard()
base = CNN1D()
base.load_state_dict(torch.load("sorties/cnn_meilleur.pt"))
base.eval()


def exactitude_torch(m):
    with torch.no_grad():
        return ((m(torch.from_numpy(Xte).unsqueeze(1)) > 0).numpy() == yte).mean()


# ---------- 1) Élagage : on met à zéro les poids les plus petits -----------------
print("== Élagage (poids mis à zéro) ==")
print(f"{'0 % élagué':<14} exactitude = {exactitude_torch(base):.3f}")
for frac in [0.3, 0.5, 0.7, 0.9]:
    m = copy.deepcopy(base)
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
            prune.l1_unstructured(mod, "weight", amount=frac)
    print(f"{int(frac * 100):>3d} % élagué   exactitude = {exactitude_torch(m):.3f}")
print("(Attention : ça ne rend PAS le calcul plus rapide tel quel, les zéros sont quand même calculés.)\n")


# ---------- 1 bis) Élagage puis réentraînement (étape 8, exercice 1) --------------
# On élague, puis on réentraîne quelques epochs. Le masque d'élagage reste en place :
# les poids supprimés restent à zéro, seuls les poids restants s'ajustent pour compenser.
EPOCHS_REPRISE = 3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).to(device)
ytr_t = torch.from_numpy(ytr).float().to(device)
print(f"== Élagage puis réentraînement ({EPOCHS_REPRISE} epochs) ==")
for frac in [0.5, 0.7, 0.9]:
    torch.manual_seed(0)
    m = copy.deepcopy(base)
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
            prune.l1_unstructured(mod, "weight", amount=frac)
    avant = exactitude_torch(m)
    m.to(device).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    perte_bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS_REPRISE):
        ordre = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(ordre), 64):
            idx = ordre[i:i + 64]
            opt.zero_grad()
            perte_bce(m(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    m.to("cpu").eval()
    print(f"{int(frac * 100):>3d} % élagué   avant : {avant:.3f}   après réentraînement : {exactitude_torch(m):.3f}")
print()


# ---------- 2) Quantification INT8 avec ONNX Runtime -----------------------------
class Calibration(CalibrationDataReader):
    """Fournit quelques signaux typiques pour que la quantification choisisse ses échelles."""
    def __init__(self, X):
        self.it = iter([{"signal": X[i:i + 1].reshape(1, 1, 512)} for i in range(200)])
    def get_next(self):
        return next(self.it, None)


quantize_static("sorties/cnn.onnx", "sorties/cnn_int8.onnx", Calibration(Xtr),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)


def evaluer_onnx(chemin):
    sess = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    X = Xte.reshape(-1, 1, 512)
    sortie = sess.run(None, {"signal": X})[0]
    exact = ((sortie > 0) == yte).mean()
    # rappel (étape 8, exercice 4) : au seuil 0,5, et au seuil d'inspection choisi à l'étape 5
    proba = 1 / (1 + np.exp(-sortie))
    rappel_05 = (proba[yte == 1] >= 0.5).mean()
    rappel_seuil = (proba[yte == 1] >= SEUIL).mean()
    rates = int((proba[yte == 1] < SEUIL).sum())
    # latence : un signal à la fois, médiane sur 300 appels
    un = X[:1]
    for _ in range(30):
        sess.run(None, {"signal": un})
    t = []
    for _ in range(300):
        t0 = time.perf_counter(); sess.run(None, {"signal": un}); t.append(time.perf_counter() - t0)
    return exact, np.median(t) * 1000, os.path.getsize(chemin) / 1024, rappel_05, rappel_seuil, rates


SEUIL = json.load(open("sorties/seuil.json"))["seuil"]   # seuil d'inspection choisi à l'étape 5
print("== Quantification INT8 (ONNX Runtime) ==")
print(f"{'modèle':<10}{'exactitude':>12}{'latence (ms)':>15}{'taille (Ko)':>13}"
      f"{'rappel 0.5':>12}{f'rappel {SEUIL}':>13}{'ratés':>8}")
for nom, chemin in [("float32", "sorties/cnn.onnx"), ("int8", "sorties/cnn_int8.onnx")]:
    e, l, s, r05, rs, rates = evaluer_onnx(chemin)
    print(f"{nom:<10}{e:>12.3f}{l:>15.3f}{s:>13.1f}{r05:>12.3f}{rs:>13.3f}{rates:>8d}")


# ---------- 3) Latence selon la taille du lot (étape 8, exercice 2) -------------
def latence_lot(chemin, taille, repetitions=200):
    """Temps médian (ms) pour traiter un lot de `taille` signaux d'un coup."""
    sess = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])
    lot = Xte[:taille].reshape(taille, 1, 512)
    for _ in range(20):                       # échauffement
        sess.run(None, {"signal": lot})
    t = []
    for _ in range(repetitions):
        t0 = time.perf_counter(); sess.run(None, {"signal": lot}); t.append(time.perf_counter() - t0)
    return np.median(t) * 1000


print("\n== Latence selon la taille du lot ==")
print(f"{'modèle':<10}{'lot':>6}{'lot entier (ms)':>18}{'par signal (ms)':>18}")
for nom, chemin in [("float32", "sorties/cnn.onnx"), ("int8", "sorties/cnn_int8.onnx")]:
    for taille in [1, 16, 128]:
        total = latence_lot(chemin, taille)
        print(f"{nom:<10}{taille:>6}{total:>18.3f}{total / taille:>18.4f}")
