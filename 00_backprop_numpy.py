"""
Un réseau de neurones minuscule, écrit à la main avec NumPy, SANS PyTorch.
Il apprend la fonction XOR. Le but : voir de ses yeux la rétropropagation.

  entrée (2) -> couche cachée (4 neurones, tanh) -> sortie (1, sigmoïde)
"""
import numpy as np

# Graine 1 : avec la graine 0 ce réseau se bloque dans un minimum local (essaie !).
rng = np.random.default_rng(1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros((1, 4))
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros((1, 1))
sigmoide = lambda z: 1 / (1 + np.exp(-z))


def avant(X, W1, b1, W2, b2):
    h = np.tanh(X @ W1 + b1)          # activations cachées
    p = sigmoide(h @ W2 + b2)         # probabilité de sortie
    return h, p


def perte(p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- Vérification du gradient : la rétropropagation est-elle juste ? -------------
def gradients(W1, b1, W2, b2):
    h, p = avant(X, W1, b1, W2, b2)
    dz2 = (p - y) / len(X)                       # dérivée de la perte par rapport à la sortie
    dW2, db2 = h.T @ dz2, dz2.sum(0, keepdims=True)
    dh = dz2 @ W2.T
    dz1 = dh * (1 - h ** 2)                      # dérivée de tanh
    dW1, db1 = X.T @ dz1, dz1.sum(0, keepdims=True)
    return dW1, db1, dW2, db2


dW1, _, _, _ = gradients(W1, b1, W2, b2)
eps, i, j = 1e-6, 0, 0
W1[i, j] += eps;  p_plus = perte(avant(X, W1, b1, W2, b2)[1])
W1[i, j] -= 2 * eps;  p_moins = perte(avant(X, W1, b1, W2, b2)[1])
W1[i, j] += eps
numerique = (p_plus - p_moins) / (2 * eps)
print(f"Gradient par rétropropagation : {dW1[i, j]:+.8f}")
print(f"Gradient par différence finie : {numerique:+.8f}   (doivent être quasi identiques)\n")

# --- Entraînement par descente de gradient ---------------------------------------
LR = 0.5
for etape in range(2001):
    dW1, db1, dW2, db2 = gradients(W1, b1, W2, b2)
    W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2
    if etape % 400 == 0:
        print(f"étape {etape:4d}  perte = {perte(avant(X, W1, b1, W2, b2)[1]):.4f}")

print("\nPrédictions finales (attendu : 0 1 1 0) :")
print(np.round(avant(X, W1, b1, W2, b2)[1].ravel(), 3))
