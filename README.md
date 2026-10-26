# Détection de défauts sur signaux ultrasonores avec un CNN 1D

Projet d'apprentissage : simuler des signaux d'inspection ultrasonore (A-scan), entraîner un réseau de neurones convolutif 1D pour détecter les défauts, évaluer, tester la robustesse, optimiser, déployer (ONNX, C++, API).

Les données sont **simulées** : le projet sert à apprendre la méthode, pas à prétendre une expertise métier.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution (dans l'ordre)

```bash
python simulateur.py           # image d'exemples de signaux
python 00_backprop_numpy.py    # rétropropagation à la main
python 01_premier_reseau.py    # premier réseau PyTorch
python 02_baseline.py          # méthodes sans réseau de neurones
python 03_train_cnn.py         # entraînement du CNN (~1 à 2 min)
python 04_evaluation.py        # métriques, seuil, matrice de confusion
python 05_robustesse.py        # bruit, étalonnage, décalage, attaque adverse
python 06_export_onnx.py       # export au format ONNX
python 07_quantification.py    # élagage et quantification INT8
python 08_api_flask.py --test  # API de diagnostic (ou sans --test pour la lancer)
cd cpp && python exporter_poids.py && g++ -O2 -std=c++17 -o infer infer.cpp && ./infer
```

## Résultats obtenus (jeu de test simulé)

| Méthode | Exactitude | AUC |
|---|---|---|
| Porte fixe avec seuil (méthode classique) | 0,65 | 0,70 |
| Régression logistique sur le signal brut | 0,51 | 0,50 |
| CNN 1D (15 169 paramètres) | 0,86 | 0,92 |

Le guide complet qui accompagne ce dépôt explique chaque étape pour un débutant.
