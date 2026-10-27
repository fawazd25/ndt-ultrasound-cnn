# Réponses aux exercices

Mes réponses aux exercices du guide, étape par étape. Ce que j'ai compris en chemin est dans `../journal.md`.

## Étape 0 : rétropropagation à la main (`00_backprop_numpy.py`)

### Exercice 1. Graine 0

À faire : lancer moi-même avec `default_rng(0)` et noter les prédictions obtenues.

D'après un test fait pendant l'exercice 2, avec 4 neurones cachés et la graine 0, la perte reste bloquée vers 0,35 et les prédictions valent `[0.002 0.499 0.999 0.501]`. Deux exemples sur quatre restent autour de 0,5 : le réseau est coincé dans un mauvais minimum, sans erreur ni message.

### Exercice 2. Couche cachée de 4 à 8 neurones

J'ai changé les deux lignes d'initialisation :

```python
W1, b1 = rng.normal(0, 1, (2, 8)), np.zeros((1, 8))
W2, b2 = rng.normal(0, 1, (8, 1)), np.zeros((1, 1))
```

Rien d'autre à modifier : `@`, `.T` et `.sum` s'adaptent aux nouvelles formes.

Avec la graine 0 et 8 neurones, la perte descend à 0,0013 et les prédictions sont `[0.001 0.998 0.999 0.002]`, toutes justes. Ça règle donc le problème pour cette graine.

Un test sur 200 graines montre que ce n'est pas propre à la graine 0 :

| Neurones cachés | Échecs sur 200 graines |
|---|---|
| 2 | 101 (environ 50 %) |
| 4 | 5 (2,5 %) |
| 8 | 0 |
| 16 | 0 |

Passer de 4 à 8 neurones rend l'apprentissage plus robuste à l'initialisation aléatoire : le taux d'échec passe de 2,5 % à 0 % sur 200 graines. Avec plus de neurones, le réseau a plus de chemins vers une solution. Ce n'est pas une garantie : d'autres réglages (taux d'apprentissage, nombre d'étapes) pourraient encore faire échouer un départ.

### Exercice 3. `LR = 5.0` puis `LR = 0.001`

À faire.

### Exercice 4. La ligne qui déplace les poids

Ligne 53 :

```python
W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2
```

Chaque poids recule d'un pas de taille `LR` dans la direction opposée à son gradient, c'est-à-dire vers le bas de la pente de la perte.

## Étape 1 : premier réseau PyTorch (`01_premier_reseau.py`)

Le jeu de test contient 450 images, donc une image vaut environ 0,0022 d'exactitude. Je garde cette échelle en tête pour ne pas surinterpréter les petites variations.

### Exercice 1. 100 epochs

L'exactitude test reste à 0,969 entre les epochs 70 et 90, puis vaut 0,971 aux epochs 95 et 100. Elle se stabilise autour de 97 %. Passer de 0,969 à 0,971, c'est une seule image de plus bien classée : le modèle a atteint un plateau.

### Exercice 2. 8 puis 128 neurones

Avec 128 neurones (`nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)`), l'exactitude finale est de 0,971, avec un pic à 0,976 en cours d'entraînement. L'écart entre le pic et la fin représente deux images, donc je ne l'interprète pas.

À faire : l'essai avec 8 neurones, puis comparer les valeurs finales pour 8, 32 et 128 neurones.

### Exercice 3. Aucune couche cachée

Ma première tentative, `nn.Sequential(nn.Linear(64, 10), nn.ReLU(), nn.Linear(10, 10))`, gardait en fait une couche cachée de 10 neurones (0,958). Le modèle demandé est `nn.Linear(64, 10)` seul.

| Modèle | Paramètres | Perte d'entraînement | Exactitude test | Erreurs sur 450 |
|---|---|---|---|---|
| `nn.Linear(64, 10)` seul | 650 | 0,324 | 0,936 | ≈ 29 |
| couche cachée de 128 | 9 610 | 0,011 | 0,971 | ≈ 13 |

Sans couche cachée, le modèle atteint déjà 93,6 % sur le test, donc ce problème est en grande partie faisable avec une simple somme pondérée des pixels. La couche cachée monte à 97,1 % et divise le nombre d'erreurs par plus de deux. Elle est utile parce qu'elle combine des pixels entre eux, ce qu'un modèle linéaire ne sait pas faire.

Le gain est bien plus faible que ce que suggère la perte d'entraînement, presque 30 fois plus basse avec la couche cachée. Cette perte est mesurée sur les images que le modèle a déjà vues : une partie de l'écart vient de ce que le plus gros modèle colle mieux à l'entraînement, et elle ne se retrouve pas sur des images nouvelles. C'est l'exactitude test qui juge la prédiction.

### Exercice 4. Sans `opt.zero_grad()`

L'exactitude test tombe à 0,536.

PyTorch additionne chaque nouveau gradient dans `.grad` au lieu de le remplacer. Sans remise à zéro, le gradient utilisé à une étape est la somme de tous les gradients depuis le début. La plupart ont été calculés à des endroits où les poids ne sont plus, donc le pas suit surtout l'ancienne pente au lieu de la pente actuelle.

Sur un exemple à une dimension (perte w², départ w = 1, pas de 0,1), w atteint presque 0, le minimum, à l'étape 3. La somme accumulée vaut pourtant encore 4,48, parce qu'elle contient toutes les anciennes pentes positives. Le pas suivant envoie w à -0,45, puis -0,81, puis -1 : le modèle dépasse le minimum et oscille au lieu de s'y arrêter.

L'accumulation est voulue. Quand un poids sert à plusieurs endroits du calcul, son gradient total est la somme de ses contributions, et on peut découper un gros lot en morceaux en appelant `backward()` sur chacun avant un seul `step()`. En contrepartie, il faut remettre les gradients à zéro à chaque étape d'entraînement normale.

## Étape 2 : le simulateur (`simulateur.py`)

### Exercice 1. Repérer les défauts à l'œil avec `difficulte=2.0`

J'ai modifié le bloc principal en `X, y = generer(6, seed=42, difficulte=2.0)`. Sur les 6 signaux, 2 sont étiquetés DÉFAUT (en haut à droite et en bas à gauche).

Ma réponse à l'œil : j'ai cru voir deux défauts sur le signal du haut à droite, et rien sur celui du bas à gauche.

En recalculant ce que le simulateur a réellement placé dans ces signaux, on obtient ceci :

| Signal | Défaut (position, amplitude) | Écho de fond (position, amplitude) | Écart-type du bruit |
|---|---|---|---|
| haut à droite | vers 351, 0,078 | vers 441, 0,37 | 0,156 |
| bas à gauche | vers 284, 0,027 | vers 380, 0,25 | 0,041 |

Un signal défectueux ne contient qu'un seul défaut. Les deux pics que je voyais en haut à droite viennent donc du bruit ou de l'écho de fond (le pic vers 441). Le vrai défaut, vers 351, a une amplitude deux fois plus petite que l'écart-type du bruit : il est invisible. En bas à gauche, la petite bosse vers 380 est l'écho de fond, et le défaut vers 284 est minuscule (0,027).

La figure `sorties/defaut_annote.png` montre le signal du haut à droite deux fois : avec le bruit, tel que je le voyais, puis sans le bruit. Sans bruit, le défaut apparaît comme une petite oscillation vers 351, bien plus faible que l'écho de fond. Avec le bruit, rien ne le distingue des pics voisins.

![Le signal DÉFAUT du haut à droite, avec et sans bruit](sorties/defaut_annote.png)

Résultat : je ne repère correctement aucun des deux défauts. À ce niveau de difficulté, un défaut a la même taille que le bruit, voire moins, et l'œil confond le bruit, l'écho de fond et le défaut. C'est ce que le CNN devra faire mieux que moi, en reconnaissant la forme de l'impulsion plutôt que sa hauteur.

### Exercice 2. Superposer 20 signaux sains normalisés

À faire.

### Exercice 3. La simplification la plus éloignée de la réalité

À faire, à écrire dans le journal.
