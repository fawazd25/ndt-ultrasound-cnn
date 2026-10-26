"""Écrit les poids du modèle et quelques signaux de test dans des fichiers texte pour le C++."""
import sys
sys.path.insert(0, "..")
import numpy as np
import torch
from simulateur import jeux_standard
from modele import CNN1D

m = CNN1D()
m.load_state_dict(torch.load("../sorties/cnn_meilleur.pt"))
m.eval()

with open("poids.txt", "w") as f:
    convs = [l for l in m.extracteur if isinstance(l, torch.nn.Conv1d)]
    f.write(f"{len(convs)}\n")
    for c in convs:
        w, b = c.weight.detach().numpy(), c.bias.detach().numpy()   # w : (sortie, entrée, noyau)
        f.write(f"{w.shape[0]} {w.shape[1]} {w.shape[2]}\n")
        f.write(" ".join(f"{v:.9g}" for v in w.ravel()) + "\n")
        f.write(" ".join(f"{v:.9g}" for v in b) + "\n")
    w, b = m.fc.weight.detach().numpy(), m.fc.bias.detach().numpy()
    f.write(f"{w.shape[1]}\n")
    f.write(" ".join(f"{v:.9g}" for v in w.ravel()) + "\n")
    f.write(f"{b[0]:.9g}\n")

_, _, (Xte, _) = jeux_standard()
X = Xte[:20]
with torch.no_grad():
    logits = m(torch.from_numpy(X).unsqueeze(1)).numpy()
with open("test_vecteurs.txt", "w") as f:
    f.write(f"{len(X)} {X.shape[1]}\n")
    for x, l in zip(X, logits):
        f.write(f"{l:.9g}\n")
        f.write(" ".join(f"{v:.9g}" for v in x) + "\n")
print("Écrit : poids.txt et test_vecteurs.txt")
