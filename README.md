# Détection de défauts sur signaux ultrasonores avec un CNN 1D

Projet d'apprentissage : simuler des signaux d'inspection ultrasonore (A-scan), entraîner un réseau de neurones convolutif 1D pour détecter les défauts, évaluer, tester la robustesse, optimiser, déployer (ONNX, C++, API).

Les données sont **simulées** : le projet sert à apprendre la méthode, pas à prétendre une expertise métier.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testé avec Python 3.14 sur un Mac Apple Silicon. Sur une connexion lente, `pip install --timeout 120 -r requirements.txt` évite les erreurs de délai dépassé.

## Exécution (dans l'ordre)

```bash
python simulateur.py           # images d'exemples de signaux
python 00_backprop_numpy.py    # rétropropagation à la main
python 01_premier_reseau.py    # premier réseau PyTorch
python 02_baseline.py          # méthodes sans réseau de neurones
python 03_train_cnn.py         # entraînement du CNN
python 04_evaluation.py        # métriques, seuil, matrice de confusion
python 05_robustesse.py        # bruit, étalonnage, décalage, attaque adverse
python 06_export_onnx.py       # export au format ONNX
python 07_quantification.py    # élagage et quantification INT8
python 08_api_flask.py --test  # API de diagnostic (ou sans --test pour la lancer)
cd cpp && python exporter_poids.py && g++ -O2 -std=c++17 -o infer infer.cpp && ./infer
```

`03_train_cnn.py` utilise automatiquement la puce graphique des Mac Apple Silicon (backend MPS de PyTorch) et revient sur le processeur ailleurs. Sur un MacBook Pro M5, 60 epochs prennent environ 20 secondes sur MPS, contre 240 secondes sur le processeur.

## Le modèle retenu

Un CNN 1D à 5 blocs [convolution, ReLU, max-pooling], avec des noyaux de 15 échantillons, une moyenne globale et une couche linéaire de décision : 40 593 paramètres (`modele.py`).

Ce choix vient d'essais comparés sur le jeu de **validation**, en changeant un réglage à la fois :

| Configuration | Paramètres | Meilleure perte val | Exactitude val |
|---|---|---|---|
| 4 blocs, noyau 9 (modèle de départ) | 15 169 | 0,332 | 0,864 |
| 4 blocs, noyau 15 | 25 201 | 0,294 | 0,877 |
| 5 blocs, noyau 15 (retenu) | 40 593 | 0,273 | 0,883 |

Chaque configuration n'a été entraînée qu'une fois : les petits écarts sont du même ordre que la variation d'un entraînement à l'autre.

## Résultats sur le jeu de test simulé

| Méthode | Exactitude | AUC |
|---|---|---|
| Porte fixe avec seuil (méthode classique) | 0,65 | 0,70 |
| Régression logistique sur le signal brut | 0,51 | 0,50 |
| CNN 1D retenu (5 blocs, noyau 15) | à mesurer (étape 5) | à mesurer (étape 5) |

La porte fixe surveille une fenêtre de temps fixe : elle rate les défauts profonds et ne peut pas s'étendre sans attraper l'écho de fond, qui change de place d'une pièce à l'autre. La régression logistique donne un poids fixe à chaque position du signal et ne sait pas reconnaître un écho « quelque part » : elle fait le hasard.

## Documentation du travail

- [`journal.md`](journal.md) : ce que j'ai compris au fil du projet, les erreurs que j'ai faites et ce qu'elles m'ont appris.
- [`reponses_exercices.md`](reponses_exercices.md) : mes réponses aux exercices de chaque étape, avec les chiffres obtenus.
