"""Optimisation : élagage (pruning) et quantification INT8. On mesure précision, taille, vitesse."""
import os, time, copy
import numpy as np
import torch
import torch.nn.utils.prune as prune
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat)
from simulateur import jeux_standard
from modele import CNN1D

(Xtr, _), _, (Xte, yte) = jeux_standard()
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
    # latence : un signal à la fois, médiane sur 300 appels
    un = X[:1]
    for _ in range(30):
        sess.run(None, {"signal": un})
    t = []
    for _ in range(300):
        t0 = time.perf_counter(); sess.run(None, {"signal": un}); t.append(time.perf_counter() - t0)
    return exact, np.median(t) * 1000, os.path.getsize(chemin) / 1024


print("== Quantification INT8 (ONNX Runtime) ==")
print(f"{'modèle':<10}{'exactitude':>12}{'latence (ms)':>15}{'taille (Ko)':>13}")
for nom, chemin in [("float32", "sorties/cnn.onnx"), ("int8", "sorties/cnn_int8.onnx")]:
    e, l, s = evaluer_onnx(chemin)
    print(f"{nom:<10}{e:>12.3f}{l:>15.3f}{s:>13.1f}")
