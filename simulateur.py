"""
Simulateur de signaux ultrasonores (A-scan) pour l'inspection d'une pièce.

Un A-scan est un signal 1D : l'amplitude de l'écho reçu en fonction du temps.
On y voit :
  - l'écho de la surface d'entrée ("front wall"), toujours présent et fort ;
  - l'écho du fond de la pièce ("back wall"), plus faible (atténuation) ;
  - si la pièce a un défaut : un écho supplémentaire entre les deux,
    et un fond un peu plus faible (le défaut "fait de l'ombre") ;
  - du bruit électronique et du "grain" (petits échos du matériau).

Ce n'est PAS de la vraie physique, c'est un modèle simplifié pour apprendre.
"""
import os
import numpy as np

# Les scripts écrivent leurs résultats dans sorties/ : on s'assure que le dossier existe.
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorties"), exist_ok=True)

LONGUEUR = 512          # nombre d'échantillons par signal
F0 = 0.12               # fréquence de l'impulsion (cycles par échantillon)
LARGEUR = 6.0           # largeur de l'enveloppe de l'impulsion
ZONE_MORTE = 100        # on ignore le début du signal (écho d'entrée)


def _impulsion(demi=24):
    t = np.arange(-demi, demi + 1)
    return np.exp(-(t / LARGEUR) ** 2) * np.cos(2 * np.pi * F0 * t)


IMP = _impulsion()


def ajouter_echo(sig, pos, amp):
    """Ajoute une impulsion d'amplitude `amp` centrée sur l'échantillon `pos`."""
    pos = int(round(pos))
    d = len(IMP) // 2
    a, b = max(0, pos - d), min(len(sig), pos + d + 1)
    if b > a:
        sig[a:b] += amp * IMP[a - (pos - d): b - (pos - d)]


def generer(n, taux_defaut=0.5, seed=0, difficulte=1.0):
    """
    Retourne X (n, LONGUEUR) float32 et y (n,) int : 1 = défaut, 0 = pièce saine.
    `difficulte` > 1 augmente le bruit et réduit l'amplitude des défauts.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, LONGUEUR), dtype=np.float32)
    y = (rng.random(n) < taux_defaut).astype(np.int64)
    for i in range(n):
        s = np.zeros(LONGUEUR)
        p_entree = rng.uniform(35, 55)
        epaisseur = rng.uniform(220, 420)
        p_fond = p_entree + epaisseur
        att = rng.uniform(0.0015, 0.004)            # atténuation par échantillon
        gain = rng.uniform(0.6, 1.6)                # réglage de l'appareil

        ajouter_echo(s, p_entree, 1.0)
        amp_fond = 0.9 * np.exp(-att * epaisseur)
        if y[i] == 1:
            amp_fond *= rng.uniform(0.6, 0.95)      # ombre du défaut
            pos = rng.uniform(p_entree + 45, p_fond - 45)
            amp = rng.uniform(0.10, 0.50) / difficulte * np.exp(-att * (pos - p_entree))
            ajouter_echo(s, pos, amp)
        ajouter_echo(s, p_fond, amp_fond)

        # grain du matériau : petits échos sans gravité
        for _ in range(rng.integers(5, 25)):
            ajouter_echo(s, rng.uniform(p_entree + 30, p_fond), rng.uniform(-0.05, 0.05))

        s = gain * s + rng.normal(0, rng.uniform(0.02, 0.08) * difficulte, LONGUEUR)
        X[i] = s
    return X, y


def normaliser(X):
    """Calibre chaque signal sur son écho d'entrée, puis masque la zone morte."""
    ref = np.abs(X[:, :90]).max(axis=1, keepdims=True)
    Xn = X / ref
    Xn[:, :ZONE_MORTE] = 0.0
    return Xn.astype(np.float32)


def jeux_standard(difficulte=1.0):
    """Jeux d'entraînement / validation / test avec des graines différentes."""
    Xtr, ytr = generer(6000, seed=1, difficulte=difficulte)
    Xva, yva = generer(2000, seed=2, difficulte=difficulte)
    Xte, yte = generer(2000, seed=3, difficulte=difficulte)
    return (normaliser(Xtr), ytr), (normaliser(Xva), yva), (normaliser(Xte), yte)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, y = generer(6, seed=42)
    fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for k, a in enumerate(ax.ravel()):
        a.plot(X[k], lw=0.8)
        a.set_title("DÉFAUT" if y[k] else "saine", color="crimson" if y[k] else "green")
    fig.supxlabel("temps (échantillons)")
    fig.supylabel("amplitude")
    plt.tight_layout()
    plt.savefig("sorties/exemples_signaux.png", dpi=110)
    print("Image écrite : sorties/exemples_signaux.png")

    # --- Exercice 2 : 20 signaux sains normalisés, superposés ---
    Xs, ys = generer(20, taux_defaut=0.0, seed=7)  # 20 signaux, aucun défaut
    Xs = normaliser(Xs)  # même traitement que pour le réseau
    plt.figure(figsize=(11, 4))
    for s in Xs:  # un tracé par signal, sur le même graphique
        plt.plot(s, lw=0.6, alpha=0.5)
    plt.title("20 signaux sains normalisés, superposés")
    plt.xlabel("temps (échantillons)")
    plt.ylabel("amplitude normalisée")
    plt.tight_layout()
    plt.savefig("sorties/signaux_sains_superposes.png", dpi=110)
    print("Image écrite : sorties/signaux_sains_superposes.png")