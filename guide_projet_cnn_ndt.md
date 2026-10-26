# Guide complet : apprendre les réseaux de neurones en détectant des défauts sur des signaux d'inspection

*Écrit pour toi, Fawaz, qui viens de la cybersécurité et découvres le machine learning. Objectif : arriver au 5 janvier 2027 avec un projet fini, compris et présentable, qui correspond aux tâches du stage chez Evident Canada (offre 501978).*

---

## Table des matières

0. Avant de commencer
1. Les idées en 30 minutes, sans code
2. Le domaine : l'inspection par ultrasons et par courants de Foucault
3. Installation
4. Le projet étape par étape (étapes 0 à 10)
5. Calendrier : 10 semaines, du 26 octobre au 3 janvier
6. Présenter le projet (GitHub, CV, entretien)
7. Ressources pour apprendre
8. Glossaire
9. Dépannage
10. Ce que ce projet ne prouve pas

---

## 0. Avant de commencer

### Ce que tu auras à la fin

Un dépôt GitHub contenant un projet complet qui :

1. simule des signaux ultrasonores de pièces saines et défectueuses ;
2. compare une méthode classique (seuil), une régression logistique et un réseau de neurones convolutif ;
3. évalue le réseau avec les bonnes métriques et choisit un seuil de décision pour un contexte où rater un défaut coûte cher ;
4. teste la robustesse du modèle (bruit, mauvais étalonnage, attaque adverse) ;
5. l'optimise (quantification INT8) et mesure le gain ;
6. l'exporte (ONNX), le fait tourner en **C++**, et l'expose derrière une **API**.

Chaque point correspond à une ligne de l'offre : concevoir et programmer un réseau de neurones, le déployer, le tester, mesurer sa performance, l'optimiser, l'intégrer à une application, en Python ou en C++.

### Comment travailler (important)

Le code complet est fourni dans le dossier `projet_ndt_cnn/`. **Ne te contente pas de le lancer.** Pour chaque étape :

1. **Lis** l'explication dans ce guide.
2. **Lance** le script et compare ta sortie à celle du guide.
3. **Relis** le code ligne par ligne jusqu'à pouvoir l'expliquer à voix haute.
4. **Modifie** quelque chose (les exercices à la fin de chaque étape) et observe ce qui change.
5. **Casse** quelque chose volontairement pour voir le message d'erreur.
6. **Note** dans un fichier `journal.md` ce que tu as compris et ce qui t'a surpris. En entretien, ce journal est une mine d'or.

Le but n'est pas d'avoir le projet. C'est de pouvoir répondre à « pourquoi as-tu fait ce choix ? » pour chaque ligne.

### Ce qui a été testé et ce qui ne l'a pas été

Tout le code a été exécuté de bout en bout sur une machine Linux avec Python 3.13, PyTorch 2.14, scikit-learn 1.9, ONNX Runtime 1.29 et g++. Les résultats donnés plus bas viennent de cette exécution. Sur ton Mac, **tes chiffres seront proches mais pas forcément identiques** (versions différentes, calculs en virgule flottante légèrement différents). Si tu trouves 0,85 au lieu de 0,86, tout va bien. Si tu trouves 0,50, il y a un problème : voir la section Dépannage. Je n'ai pas pu tester sur macOS.

### Temps nécessaire

Compte 8 à 10 heures par semaine pendant 10 semaines. Le calendrier de la section 5 tient compte du fait que tu as aussi des cours et des examens en décembre.

---

## 1. Les idées en 30 minutes, sans code

Lis cette section d'abord, lentement. Tout le reste en dépend.

### 1.1 Programmer à la main contre apprendre à partir d'exemples

En programmation classique, tu écris les règles : « si l'écho dans cette fenêtre dépasse ce seuil, alors alarme ». Tu connais le problème, tu codes la solution.

En apprentissage automatique (machine learning), tu fournis des **exemples** (des signaux et leur étiquette : « défaut » ou « saine ») et un programme **trouve lui-même les règles**. C'est utile quand les règles sont trop nombreuses ou trop subtiles pour être écrites à la main : un écho faible noyé dans du bruit, qui apparaît à une profondeur variable, sur des pièces d'épaisseurs différentes.

Un **modèle** est le résultat de cet apprentissage : une fonction qui prend un signal en entrée et renvoie une réponse (ici, une probabilité de défaut).

### 1.2 Le neurone artificiel

Un neurone fait trois choses :

1. il prend des nombres en entrée (x₁, x₂, x₃...) ;
2. il calcule une **somme pondérée** : `z = w₁·x₁ + w₂·x₂ + w₃·x₃ + b` ;
3. il applique une **fonction d'activation** à z.

Les `w` sont les **poids** et `b` est le **biais**. Ce sont les nombres que l'apprentissage va ajuster. Un réseau de neurones, c'est uniquement des milliers de ces petits calculs enchaînés. Il n'y a aucune magie.

La fonction d'activation la plus courante s'appelle **ReLU** : `ReLU(z) = max(0, z)`. Elle laisse passer les valeurs positives et met les négatives à zéro. Pourquoi en mettre une ? Sans elle, enchaîner des couches reviendrait à faire une seule grosse opération linéaire, et le réseau ne pourrait apprendre que des relations simples. La non-linéarité lui permet d'apprendre des formes complexes.

### 1.3 Les couches

On range les neurones en **couches** : les sorties d'une couche deviennent les entrées de la suivante. Un réseau qui a plusieurs couches est dit **profond** (deep learning).

Le nombre de poids et de biais d'un réseau s'appelle son nombre de **paramètres**. Le réseau de ce projet en a 15 169. GPT et ses semblables en ont des centaines de milliards. Ça ne change pas le principe.

### 1.4 La perte : mesurer l'erreur

Pour apprendre, le réseau doit savoir à quel point il se trompe. On définit une **fonction de perte** (loss) : un nombre qui vaut 0 quand le modèle est parfait et qui grandit avec ses erreurs.

Pour de la classification (défaut ou pas), on utilise l'**entropie croisée binaire**. Intuition : si le modèle dit « 99 % de chance de défaut » et que c'était sain, la perte est énorme. S'il dit « 60 % » et que c'était sain, elle est modérée. S'il dit « 5 % », elle est faible.

### 1.5 La descente de gradient : apprendre en descendant

Imagine la perte comme un paysage vallonné où chaque position correspond à un choix de tous les poids. Le but est d'atteindre le fond d'une vallée (perte faible). Tu es dans le brouillard et tu ne vois que la pente sous tes pieds.

Le **gradient** donne cette pente : pour chaque poids, de combien la perte augmenterait si on augmentait un peu ce poids. L'algorithme est donc :

1. calculer le gradient de la perte par rapport à chaque poids ;
2. déplacer chaque poids un petit peu **dans la direction opposée** au gradient ;
3. recommencer.

La taille du pas s'appelle le **taux d'apprentissage** (learning rate, `lr`). Trop grand : tu sautes par-dessus la vallée et ça diverge. Trop petit : ça n'avance pas. Ici on utilise **Adam**, une variante qui ajuste le pas automatiquement pour chaque poids.

### 1.6 La rétropropagation : calculer tous les gradients efficacement

Il y a des milliers de poids. Calculer le gradient de chacun séparément serait lent. La **rétropropagation** (backpropagation) utilise la règle de dérivation en chaîne pour les obtenir tous en un seul passage en arrière, de la sortie vers l'entrée.

Tu n'as pas besoin de la dériver toi-même. Mais tu dois **l'avoir vue fonctionner une fois**, c'est le but de l'étape 0 : tu y écris un mini-réseau à la main, et tu vérifies que ton gradient correspond à une estimation numérique indépendante.

PyTorch calcule les gradients automatiquement (`loss.backward()`). Mais quand le modèle donne de mauvais résultats, comprendre ce qui se passe dessous fait la différence.

### 1.7 Entraînement et inférence

- **Entraînement** : on ajuste les poids avec beaucoup d'exemples. C'est long et coûteux.
- **Inférence** : les poids sont figés, on utilise le modèle pour prédire sur un nouveau signal. C'est rapide.

Dans une application industrielle, on entraîne une fois (sur un gros ordinateur), puis on **déploie** pour l'inférence (parfois sur un petit appareil). Le stage décrit ces deux côtés : améliorer l'environnement d'entraînement et déployer.

### 1.8 Trois jeux de données, trois rôles

C'est la règle la plus importante de toute la discipline.

| Jeu | Rôle | Tu peux le regarder ? |
|---|---|---|
| **Entraînement** (train) | Le modèle apprend dessus | Oui, à volonté |
| **Validation** | Tu t'en sers pour choisir les réglages (nombre d'epochs, seuil, architecture) | Oui, pendant le développement |
| **Test** | Note finale honnête, une seule fois à la fin | **Non** avant la fin |

Pourquoi ? Si tu règles ton modèle jusqu'à ce qu'il soit bon sur le jeu de test, tu l'as indirectement entraîné dessus et ton score est trop optimiste. Dans ce projet, le seuil de décision est choisi sur la validation, puis évalué sur le test. C'est exprès.

Une **epoch** est un passage complet sur tout le jeu d'entraînement.

### 1.9 Le surapprentissage

Un modèle peut apprendre le jeu d'entraînement par cœur (y compris son bruit) et échouer sur des données nouvelles. Signal d'alerte : la perte d'entraînement continue de baisser pendant que la perte de validation remonte. Dans ce projet, on **sauvegarde le modèle de l'epoch où la perte de validation est la plus basse**, pas le dernier.

### 1.10 Pourquoi un réseau convolutif pour un signal

Un signal d'inspection est une suite de mesures dans le temps. Un défaut se manifeste comme un **petit motif local** (un écho). Peu importe qu'il soit à l'échantillon 200 ou 300, c'est la même forme.

Une **convolution** fait glisser une petite fenêtre (un **filtre** ou **noyau**, ici de 9 échantillons) le long du signal. À chaque position, elle calcule une somme pondérée. Le même filtre est utilisé partout, donc le réseau apprend un détecteur de motif qui fonctionne à n'importe quelle position, avec très peu de paramètres. Les filtres sont les poids appris : l'entraînement trouve tout seul quelles formes méritent d'être détectées.

Le **max-pooling** réduit la longueur du signal de moitié en gardant le plus grand de chaque paire. Ça condense l'information et élargit ce que chaque neurone « voit » (son **champ réceptif**) d'une couche à l'autre. Les premières couches détectent des oscillations, les suivantes des échos entiers, les dernières des échos dans leur contexte.

---

## 2. Le domaine : l'inspection par ultrasons et par courants de Foucault

Evident fabrique des équipements de **contrôle non destructif** (CND, ou NDT en anglais) : vérifier l'intérieur d'une pièce (soudure, tuyau, aile d'avion) sans l'abîmer. Je simplifie volontairement. Pour les détails, voir les ressources en section 7.

### 2.1 Les ultrasons

Une sonde envoie une impulsion sonore très aiguë dans la pièce, puis écoute les échos. À chaque changement de milieu (surface, défaut, fond de la pièce), une partie de l'onde est renvoyée. Le temps entre l'envoi et l'écho indique la **profondeur** (le son va à une vitesse connue dans le matériau). L'amplitude de l'écho indique la **taille ou la nature de ce qui l'a renvoyé**.

Le signal enregistré en un point s'appelle un **A-scan** : l'amplitude en fonction du temps. C'est ce que simule ce projet :

- **Écho d'entrée** (front wall) : la surface de la pièce. Toujours présent, toujours fort.
- **Écho de fond** (back wall) : l'autre face de la pièce. Plus faible, car le son s'atténue en voyageant.
- **Écho de défaut** : s'il y a une fissure ou une inclusion entre les deux, un écho supplémentaire apparaît, et le fond devient plus faible (le défaut fait de l'ombre).
- **Bruit** : électronique de l'appareil, et « grain » du matériau (de multiples petits échos sans gravité).

Le défi : un défaut peu profond donne un écho faible, à une position inconnue (car l'épaisseur varie), parmi du bruit et du grain. C'est exactement le type de tâche où un seuil fixe est limité et où un réseau peut aider.

### 2.2 Concepts de métier que tu retrouves dans le code

| Concept | Où c'est dans le projet |
|---|---|
| **Zone morte** : juste après l'écho d'entrée, l'appareil est « ébloui » et on ne peut rien interpréter | `ZONE_MORTE = 100` dans `simulateur.py` : on met le début à zéro |
| **Étalonnage** : régler l'appareil sur un étalon de référence | `normaliser()` : chaque signal est divisé par l'amplitude de son écho d'entrée |
| **Porte** (gate) : fenêtre de temps où l'on surveille les échos, avec un seuil | `02_baseline.py`, la méthode classique |
| **Atténuation** : l'onde perd de l'énergie avec la distance | `att` dans le simulateur |

### 2.3 Les courants de Foucault (eddy current)

Une bobine parcourue par un courant alternatif crée un champ magnétique, qui induit des courants dans une pièce métallique proche. Un défaut (fissure, corrosion) perturbe ces courants, ce qui modifie l'**impédance** de la bobine. On mesure cette variation. Le signal est souvent représenté comme une courbe dans un plan à deux dimensions (le « plan d'impédance »).

Ce projet se concentre sur les ultrasons parce que c'est plus simple à simuler. Les idées apprises (CNN, évaluation, déploiement) se transposent. En entretien, dis simplement que tu as choisi les ultrasons comme terrain d'apprentissage et que tu sais que les courants de Foucault ont leurs propres signaux, que tu aimerais apprendre.

---

## 3. Installation

### 3.1 Récupérer le projet

Dézippe `projet_ndt_cnn.zip` dans `Documents/Stage/`. Tu obtiens :

```
projet_ndt_cnn/
├── README.md
├── requirements.txt
├── simulateur.py         données simulées
├── modele.py             le réseau de neurones
├── 00_backprop_numpy.py
├── 01_premier_reseau.py
├── 02_baseline.py
├── 03_train_cnn.py
├── 04_evaluation.py
├── 05_robustesse.py
├── 06_export_onnx.py
├── 07_quantification.py
├── 08_api_flask.py
├── cpp/
│   ├── exporter_poids.py
│   └── infer.cpp
└── sorties/              créé et rempli par les scripts
```

### 3.2 Environnement Python

Dans le Terminal :

```bash
cd ~/Documents/Stage/projet_ndt_cnn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Un **environnement virtuel** (`.venv`) isole les bibliothèques du projet de celles du reste de ton Mac. À chaque nouvelle session de Terminal, retourne dans le dossier et refais `source .venv/bin/activate`. Le début de ta ligne de commande affiche alors `(.venv)`.

L'installation de PyTorch télécharge plusieurs centaines de Mo. Laisse-la finir.

Vérifie que tout fonctionne :

```bash
python -c "import torch, sklearn, onnxruntime, flask; print('ok', torch.__version__)"
```

### 3.3 Compilateur C++ (pour l'étape 9)

Sur Mac : `xcode-select --install` (installe les outils de développement d'Apple). Ensuite `g++ --version` doit répondre quelque chose. Sur Mac, `g++` est en réalité le compilateur `clang`, ça convient.

### 3.4 Git

Initialise un dépôt dès maintenant et fais un commit après chaque étape terminée. Un historique de commits réguliers prouve que tu as travaillé progressivement.

```bash
git init
printf ".venv/\n__pycache__/\nsorties/*.pt\nsorties/*.onnx\n" > .gitignore
git add . && git commit -m "Départ du projet"
```

---

## 4. Le projet étape par étape

Chaque étape suit le même plan : **objectif, lancer, ce que tu dois voir, comprendre, exercices**.

---

### Étape 0. Rétropropagation à la main (`00_backprop_numpy.py`)

**Objectif.** Voir de tes yeux ce qu'est un gradient et vérifier que la rétropropagation est juste, avec NumPy uniquement.

**Lancer.**

```bash
python 00_backprop_numpy.py
```

**Ce que tu dois voir.**

```
Gradient par rétropropagation : +0.04450988
Gradient par différence finie : +0.04450988   (doivent être quasi identiques)

étape    0  perte = ...
...
Prédictions finales (attendu : 0 1 1 0) :
[0.    0.997 0.997 0.004]
```

**Comprendre.** Le réseau apprend la fonction XOR (vrai si exactement une des deux entrées vaut 1). Un seul neurone n'y arrive pas, c'est le cas classique qui justifie les couches cachées.

Dans le code, trois blocs :

1. `avant(...)` : la **passe avant**. Les entrées traversent les deux couches et sortent en probabilité.
2. `gradients(...)` : la **passe arrière**. On part de l'erreur à la sortie (`p - y`) et on remonte, couche par couche, en multipliant par les dérivées locales (celle de tanh est `1 - h²`). Chaque ligne est une application de la règle de dérivation en chaîne.
3. La **vérification numérique** : on déplace un poids de ±0,000001, on regarde de combien la perte change, on divise. C'est une estimation lente mais indépendante du gradient. Les deux nombres affichés doivent coïncider. Cette technique (gradient checking) sert à détecter des bugs dans une rétropropagation écrite à la main.

**Exercices.**

1. Remplace `default_rng(1)` par `default_rng(0)` et relance. Avec cette graine, deux des quatre prédictions restent bloquées autour de 0,5 : le réseau est coincé dans un mauvais minimum. Conclusion à retenir : **l'initialisation aléatoire compte**, et un entraînement peut échouer sans erreur ni message.
2. Passe la couche cachée de 4 à 8 neurones. Est-ce que ça règle le problème de la graine 0 ?
3. Mets `LR = 5.0`, puis `LR = 0.001`. Décris ce qui se passe dans les deux cas et relie-le à la section 1.5.
4. Dans le code, repère la ligne qui correspond à « déplacer chaque poids dans la direction opposée au gradient ».

---

### Étape 1. Premier réseau avec PyTorch (`01_premier_reseau.py`)

**Objectif.** Découvrir le squelette PyTorch sur un problème simple : reconnaître des chiffres manuscrits (images de 8×8 pixels fournies avec scikit-learn, donc pas de téléchargement).

**Lancer.**

```bash
python 01_premier_reseau.py
```

**Ce que tu dois voir.** L'exactitude sur le jeu de test monte de 0,74 (epoch 5) à environ 0,92 (epoch 30).

**Comprendre.** C'est le squelette de **tous** les entraînements PyTorch :

```python
modele = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))  # 1. le modèle
opt = torch.optim.Adam(modele.parameters(), lr=1e-2)                      # 2. l'optimiseur
perte = nn.CrossEntropyLoss()                                             # 3. la perte

for epoch in range(...):
    opt.zero_grad()                  # remet les gradients à zéro (sinon ils s'accumulent)
    l = perte(modele(Xtr), ytr)      # passe avant + calcul de l'erreur
    l.backward()                     # rétropropagation : calcule tous les gradients
    opt.step()                       # met à jour les poids
```

Apprends ces cinq lignes par cœur. Elles réapparaissent à l'étape 4 presque à l'identique.

Détails à connaître : `Linear(64, 32)` est une couche de 32 neurones qui reçoivent chacun les 64 pixels. `CrossEntropyLoss` est l'entropie croisée pour 10 classes. Ici, tout le jeu d'entraînement passe en un seul lot à chaque epoch (le plus simple). À l'étape 4, on utilisera des **mini-lots**.

**Exercices.**

1. Passe à 100 epochs. L'exactitude continue-t-elle de monter ? Se stabilise-t-elle ?
2. Change la couche cachée : 8 neurones, puis 128. Compare.
3. Remplace le modèle par `nn.Linear(64, 10)` seul (aucune couche cachée). Compare. Que t'apprend ce résultat sur l'utilité des couches cachées pour *ce* problème ?
4. Retire `opt.zero_grad()` et relance. Que se passe-t-il, et pourquoi ?

---

### Étape 2. Les données : le simulateur (`simulateur.py`)

**Objectif.** Comprendre ce qu'on demande au réseau de détecter.

**Lancer.**

```bash
python simulateur.py
```

Ouvre `sorties/exemples_signaux.png`. Tu vois six signaux, étiquetés « saine » ou « DÉFAUT ».

**Comprendre.** Sur chaque graphique :

- le grand pic près de l'échantillon 50 est l'écho d'entrée ;
- un pic plus petit, plus à droite, est l'écho de fond (sa position dépend de l'épaisseur de la pièce) ;
- sur un signal défectueux, un troisième petit pic se glisse entre les deux ;
- tout est noyé dans du bruit.

Regarde bien : **à l'œil, certains défauts sont presque invisibles**. C'est voulu. Un signal sain a aussi des petits pics (le grain), et un signal sain court a son fond à l'échantillon 350 alors qu'un signal long l'a à 450. Ni la position ni l'amplitude ne suffisent seules.

Dans `generer()`, repère :

- `epaisseur`, `att`, `gain` : les sources de variabilité d'une pièce à l'autre ;
- `y[i] == 1` : le bloc qui ajoute le défaut et l'ombre sur le fond ;
- le grain, avec `for _ in range(rng.integers(5, 25))` ;
- `rng.normal(...)` : le bruit.

Dans `normaliser()` : division par l'écho d'entrée (étalonnage) puis mise à zéro de la zone morte.

Les signaux sont reproductibles grâce à la graine (`seed`). Le jeu d'entraînement, de validation et de test utilisent des graines différentes (1, 2, 3), donc des signaux différents.

**Exercices.**

1. Dans le bloc `if __name__ == "__main__":`, appelle `generer(6, seed=42, difficulte=2.0)` : tu augmentes le bruit et tu diminues l'amplitude des défauts. Combien de défauts parviens-tu encore à repérer à l'œil ?
2. Ajoute un graphique qui superpose 20 signaux sains normalisés. Qu'est-ce qui varie le plus ?
3. Dans le simulateur, quelle simplification te paraît la plus éloignée de la réalité ? Écris-la dans ton journal. Cette réflexion te servira dans la section 10.

---

### Étape 3. Les méthodes de référence (`02_baseline.py`)

**Objectif.** Savoir ce qu'un réseau doit battre. **Règle d'or : toujours commencer par une méthode simple.** Si elle suffit, inutile de compliquer. Et si ton réseau ne la bat pas, tu as un problème.

**Lancer.**

```bash
python 02_baseline.py
```

**Ce que tu dois voir.**

```
[porte fixe]  seuil=0.171  exactitude=0.653  rappel=0.607  précision=0.673  AUC=0.704
[logistique]  exactitude=0.505  rappel=0.471  précision=0.510  AUC=0.502
```

**Comprendre.**

- **Porte fixe** : on surveille la fenêtre 100 à 230 et on déclenche l'alarme si un écho dépasse un seuil. C'est l'équivalent d'un réglage de technicien. Le seuil est choisi sur la validation. Résultat : 65 %. Pourquoi si peu ? Les défauts situés après l'échantillon 230 échappent à la fenêtre, et le grain déclenche de fausses alarmes.
- **Régression logistique** sur le signal brut : 50,5 %, donc **le hasard**. Un modèle linéaire sur les échantillons bruts ne sait pas dire « il y a un petit écho *quelque part* », car il attribue un poids fixe à chaque position. C'est précisément le problème que les convolutions résolvent.

**L'AUC** (aire sous la courbe ROC) vaut 0,5 pour le hasard et 1,0 pour un modèle parfait. On y revient à l'étape 5.

**Exercices.**

1. Change la fenêtre de la porte (`G0, G1`). Trouve-t-elle une meilleure valeur ? Pourquoi ne peut-on pas simplement la mettre sur tout le signal ?
2. Remplace le score de porte par la **somme** des valeurs absolues dans la fenêtre au lieu du maximum. Mieux ou moins bien ?
3. Pourquoi a-t-on choisi le seuil sur la validation plutôt que sur le test ?

---

### Étape 4. Le réseau de neurones convolutif (`modele.py` et `03_train_cnn.py`)

**Objectif.** Concevoir, entraîner, et suivre l'apprentissage d'un CNN 1D.

#### L'architecture

```
entrée          (1, 512)
conv 8 filtres  → ReLU → pooling   (8, 256)
conv 16 filtres → ReLU → pooling   (16, 128)
conv 32 filtres → ReLU → pooling   (32, 64)
conv 32 filtres → ReLU → pooling   (32, 32)
moyenne sur le temps               (32)
couche linéaire                    (1)  ← le "logit"
```

Lis les dimensions entre parenthèses : (nombre de canaux, longueur). À chaque bloc, le nombre de canaux monte (on détecte plus de motifs différents) et la longueur est divisée par deux (on condense dans le temps).

**Compte des paramètres** (vérifie-le, ça entraîne la lecture des couches) : une convolution a `c_sortie × c_entrée × noyau` poids plus `c_sortie` biais.

| Couche | Calcul | Paramètres |
|---|---|---|
| conv 1 | 8 × 1 × 9 + 8 | 80 |
| conv 2 | 16 × 8 × 9 + 16 | 1 168 |
| conv 3 | 32 × 16 × 9 + 32 | 4 640 |
| conv 4 | 32 × 32 × 9 + 32 | 9 248 |
| linéaire | 32 + 1 | 33 |
| **Total** | | **15 169** |

Le script affiche bien `Nombre de paramètres : 15169`.

**Le logit.** La sortie du réseau est un nombre quelconque (le *logit*). Pour obtenir une probabilité, on applique la fonction **sigmoïde** : `p = 1 / (1 + e^(-logit))`. Un logit de 0 donne 50 %, un logit positif donne plus de 50 %. La perte `BCEWithLogitsLoss` applique la sigmoïde en interne de façon numériquement stable, c'est pourquoi le modèle ne la contient pas.

**La moyenne globale** à la fin (`x.mean(dim=-1)`) fait que le réseau décide en fonction de « combien de motifs détectés, au total, sur tout le signal ». Elle rend le réseau insensible à la position exacte des échos.

#### L'entraînement

**Lancer.**

```bash
python 03_train_cnn.py
```

Ça prend environ 1 à 2 minutes sur un ordinateur ordinaire. Tu peux aussi passer un nombre d'epochs : `python 03_train_cnn.py 100`.

**Ce que tu dois voir** (extrait) :

```
epoch  1  perte train 0.6934  perte val 0.6924  acc val 0.501
epoch 15  perte train 0.5711  perte val 0.5650  acc val 0.702
epoch 31  perte train 0.4412  perte val 0.4100  acc val 0.831
...
epoch 60  perte train 0.3834  perte val 0.4417  acc val 0.805   (≈ 0,85 à d'autres epochs)
```

Au départ la perte vaut environ 0,693, ce qui est `ln 2`: le réseau devine 50/50, il n'a encore rien appris. Puis elle descend. L'exactitude de validation monte de 50 % à environ 85 %. Elle est un peu instable d'une epoch à l'autre, c'est normal avec des mini-lots.

Le script écrit `sorties/courbes_entrainement.png` : les pertes d'entraînement et de validation, et l'exactitude. Ouvre-la. Deux choses à remarquer. D'abord, la perte de validation passe souvent **sous** celle d'entraînement : c'est normal ici, car la perte d'entraînement est une moyenne calculée *pendant* que les poids changent, alors que la validation est mesurée après la mise à jour. Ensuite, regarde la toute dernière epoch : la validation peut remonter brutalement. C'est précisément pour ça qu'on garde le modèle de la **meilleure** epoch et non de la dernière.

**Ce qu'il faut comprendre dans la boucle.**

```python
ordre = torch.randperm(len(Xtr_t))              # on mélange à chaque epoch
for i in range(0, len(ordre), BATCH):           # on découpe en mini-lots de 64
    idx = ordre[i:i + BATCH]
    opt.zero_grad()
    l = perte(modele(Xtr_t[idx]), ytr_t[idx])
    l.backward()
    opt.step()
```

Un **mini-lot** (batch) est un petit paquet de 64 signaux. On met à jour les poids après chacun plutôt qu'après tout le jeu : c'est plus rapide et le bruit de l'estimation aide à ne pas rester coincé.

Après chaque epoch :

```python
modele.eval()                      # mode évaluation
with torch.no_grad():              # on ne calcule pas de gradients : inutile et plus lent
    ...
if lv < meilleur: torch.save(...)  # on garde le meilleur modèle selon la validation
```

`modele.eval()` et `torch.no_grad()` sont des réflexes à avoir quand tu n'entraînes pas.

**Exercices.**

1. Change `canaux=(8, 16, 32, 32)` dans `modele.py` : essaie `(4, 8, 16, 16)` puis `(16, 32, 64, 64)`. Note les paramètres, le temps, la perte de validation.
2. Change la taille du noyau (`noyau=5`, `noyau=15`). Quel effet sur le champ réceptif ?
3. Teste `LR = 1e-2` et `LR = 1e-4` dans `03_train_cnn.py`.
4. **Observe le surapprentissage.** Dans `simulateur.jeux_standard`, réduis le jeu d'entraînement à 300 signaux (`generer(300, seed=1, ...)`) et entraîne 100 epochs. Regarde les courbes : la perte d'entraînement doit continuer à baisser pendant que celle de validation remonte.
5. Ajoute un cinquième bloc de convolution. Est-ce que ça aide, ou est-ce que ça n'apporte rien ?
6. Remplace la moyenne globale par un maximum global (`x.amax(dim=-1)`). Prédis d'abord ce qui va se passer, puis teste. *(Indice : quel est le plus grand écho du signal, même après la zone morte ?)*

---

### Étape 5. L'évaluation (`04_evaluation.py`)

**Objectif.** Mesurer honnêtement la performance, et choisir le seuil de décision en fonction du coût des erreurs.

**Lancer.**

```bash
python 04_evaluation.py
```

**Ce que tu dois voir.**

```
AUC (test) = 0.922

Seuil standard 0.5       seuil=0.50  rappel=0.845  précision=0.865  fausses alarmes=0.134  (défauts ratés : 156/1008)
Seuil choisi pour rappel >= 95%  seuil=0.24  rappel=0.949  précision=0.693  fausses alarmes=0.426  (défauts ratés : 51/1008)
```

Ouvre `sorties/evaluation.png` : la courbe ROC et la matrice de confusion.

**Les quatre cases de la matrice de confusion.**

|  | Prédit « saine » | Prédit « défaut » |
|---|---|---|
| **Réellement saine** | VN (vrai négatif) | **FP** (fausse alarme) |
| **Réellement défaut** | **FN** (défaut raté) | VP (vrai positif) |

**Les métriques.**

- **Exactitude** (accuracy) = bonnes réponses / total. Trompeuse quand les classes sont déséquilibrées (si 1 % des pièces sont défectueuses, répondre toujours « saine » donne 99 %).
- **Rappel** (recall) = VP / (VP + FN) : *parmi les vrais défauts, quelle part détecte-t-on ?* C'est la métrique de sécurité.
- **Précision** = VP / (VP + FP) : *parmi les alarmes, quelle part est justifiée ?*
- **Taux de fausses alarmes** = FP / (FP + VN).
- **AUC** : mesure la qualité du classement des signaux, indépendamment du seuil. C'est la probabilité qu'un signal défectueux tiré au hasard reçoive un score plus élevé qu'un signal sain tiré au hasard. 0,5 = hasard, 1 = parfait. Ici 0,92.

**Le seuil, c'est une décision métier.** Le modèle sort une probabilité. À partir de quelle probabilité déclenche-t-on l'alarme ? Avec 0,5 on rate 156 défauts sur 1008 (15 %). Dans une inspection de sécurité, c'est inacceptable. On a donc cherché, **sur la validation**, le seuil le plus haut qui garantit un rappel d'au moins 95 %. Résultat : 0,24. Sur le test, le rappel est bien de 94,9 %, mais **le prix à payer est lourd** : 42,6 % de fausses alarmes.

Pas de seuil gratuit : c'est la courbe ROC qui montre cet échange. Pour monter le rappel on accepte plus de fausses alarmes.

**Attention, point crucial de réalisme.** Dans ce jeu, la moitié des pièces ont un défaut (par construction). En production, les défauts sont rares. Imagine 1 % de pièces défectueuses avec ce même modèle (rappel 0,949, fausses alarmes 0,426). Pour 100 pièces : environ 0,95 défaut détecté et 42 fausses alarmes. **Seule une alarme sur 45 est justifiée** (précision d'environ 2 %). Un modèle qui semble correct peut être inutilisable quand l'événement recherché est rare. Savoir le dire en entretien montre que tu comprends l'évaluation.

**Exercices.**

1. Change `CIBLE` à 0,99. Que deviennent les fausses alarmes ?
2. Calcule à la main, à partir des chiffres de la matrice, le rappel, la précision et le taux de fausses alarmes à seuil 0,5. Vérifie avec le script.
3. Ajoute dans le script un **intervalle de confiance** sur l'exactitude du test par bootstrap : tire 1000 fois 2000 signaux **avec remise** parmi les prédictions, calcule l'exactitude à chaque fois, prends les percentiles 2,5 % et 97,5 %. (C'est le sujet du module 3 de ton cours de maths.) Tu devrais trouver une incertitude d'environ ±1,5 point.
4. Pourquoi choisir le seuil sur la validation plutôt que sur le test ? Réponds en une phrase.

---

### Étape 6. La robustesse (`05_robustesse.py`)

**Objectif.** Vérifier que le modèle tient quand les conditions changent. C'est aussi là que ta formation en cybersécurité te donne un avantage sur la plupart des candidats.

**Lancer.**

```bash
python 05_robustesse.py
```

**Ce que tu dois voir.**

| Perturbation | Exactitude | Rappel |
|---|---|---|
| aucune | 0,856 | 0,845 |
| bruit +0,02 | 0,839 | 0,824 |
| bruit +0,05 | 0,774 | 0,802 |
| bruit +0,1 | 0,612 | 0,796 |
| bruit +0,2 | 0,512 | 0,188 |
| amplitude ×0,7 (mauvais étalonnage) | 0,801 | 0,738 |
| amplitude ×1,3 | 0,818 | 0,805 |
| décalage de −15 échantillons | 0,834 | 0,818 |
| décalage de +15 échantillons | 0,850 | 0,814 |
| attaque adverse ε = 0,01 | 0,469 | 0,550 |
| attaque adverse ε = 0,02 | 0,209 | 0,324 |
| attaque adverse ε = 0,05 | 0,131 | 0,248 |

**Comprendre.**

- **Bruit** : le modèle supporte un peu de bruit en plus, puis s'effondre. À +0,2 il est au hasard et rate presque tous les défauts.
- **Étalonnage** : une erreur de 30 % sur l'amplitude fait perdre 5 points d'exactitude. Un appareil mal étalonné a donc un effet réel.
- **Décalage** : le modèle y est peu sensible (c'est l'avantage des convolutions).
- **Attaque adverse (FGSM)** : pour chaque signal, on calcule dans quelle direction un petit changement augmente le plus l'erreur du modèle (le gradient de la perte par rapport à *l'entrée*), et on déplace le signal d'un pas ε dans cette direction. Avec ε = 0,02 (2 % de l'amplitude de l'écho d'entrée, invisible à l'œil), l'exactitude tombe à 21 %. **C'est pire que le hasard** : le modèle est confiant et faux. Un réseau peut donc être très fragile à des perturbations qu'un humain ne remarquerait pas.

**Pourquoi c'est important pour toi.** Ces attaques (*adversarial examples*) sont documentées dans le référentiel MITRE ATLAS et dans le Top 10 OWASP pour les LLM, que tu as déjà étudiés. Ici tu les appliques à un modèle industriel, pas à un chatbot. En entretien : « j'ai mesuré la robustesse de mon modèle à des perturbations réalistes (bruit, étalonnage, décalage) et à une attaque adverse ».

**Exercices.**

1. **Augmentation de données** : dans `03_train_cnn.py`, ajoute au lot d'entraînement un bruit gaussien aléatoire et un gain aléatoire avant chaque passe avant. Réentraîne, relance `05_robustesse.py` et compare. Ne devine pas, mesure.
2. Ajoute à `05_robustesse.py` l'attaque en gardant la zone morte à zéro. Est-ce que ça change quelque chose ?
3. Cherche « adversarial training » et explique en trois phrases comment ça se défendrait. Essaie ensuite de l'implémenter.
4. Dans un système réel, qui pourrait mettre en œuvre une attaque adverse sur un signal d'inspection, et à quelle condition ? Écris ton raisonnement de menace dans ton journal (qui, où, accès nécessaire).

---

### Étape 7. L'export ONNX (`06_export_onnx.py`)

**Objectif.** Sortir le modèle de PyTorch pour qu'il puisse tourner ailleurs (autre langage, autre machine).

**Lancer.**

```bash
python 06_export_onnx.py
```

**Ce que tu dois voir.**

```
Écrit : sorties/cnn.onnx
Écart max PyTorch vs ONNX Runtime : 6.91e-06
```

Un avertissement `DeprecationWarning` peut s'afficher au sujet de l'ancien exporteur : il est sans conséquence.

**Comprendre.** **ONNX** est un format d'échange : un fichier qui décrit le réseau (ses couches et ses poids) indépendamment de PyTorch. **ONNX Runtime** est un moteur d'inférence rapide qui lit ce fichier. On vérifie que les deux donnent les mêmes sorties (un écart de 7 millionièmes, dû aux arrondis). **Vérifier l'équivalence après chaque conversion est un réflexe indispensable** : un export qui marche sans erreur peut quand même donner des résultats différents.

`dynamic_axes` signifie que la taille du lot peut varier à l'exécution (1 signal ou 100).

Si un jour l'export échoue après une mise à jour de PyTorch, retire l'argument `dynamo=False` de `torch.onnx.export` et installe `pip install onnxscript`.

**Exercices.**

1. Ouvre `sorties/cnn.onnx` dans l'outil gratuit Netron (netron.app) et retrouve les couches de ton modèle.
2. Que se passe-t-il si tu donnes à ONNX Runtime un signal de 500 échantillons au lieu de 512 ? Lis l'erreur.

---

### Étape 8. L'optimisation (`07_quantification.py`)

**Objectif.** Rendre le modèle plus petit et plus rapide, et mesurer ce que ça coûte.

**Lancer.**

```bash
python 07_quantification.py
```

**Ce que tu dois voir.**

*Élagage*

| Part des poids mis à zéro | Exactitude |
|---|---|
| 0 % | 0,856 |
| 30 % | 0,606 |
| 50 % | 0,502 |
| 70 % et plus | 0,504 |

*Quantification INT8*

| Modèle | Exactitude | Latence (ms) | Taille (Ko) |
|---|---|---|---|
| float32 | 0,856 | environ 0,07 à 0,10 | 62,3 |
| int8 | 0,846 | environ 0,04 à 0,05 | 28,1 |

La latence varie d'une exécution à l'autre selon la machine ; les ordres de grandeur et le rapport (environ 1,8 fois plus rapide) restent.

**Comprendre.**

- **Élagage (pruning)** : on met à zéro les poids de plus petite valeur absolue. Sur ce **petit** modèle, **sans réentraînement**, supprimer 30 % des poids détruit les performances. Ce n'est pas un bug : un réseau de 15 000 paramètres n'a presque pas de redondance. En pratique, on élague *puis on réentraîne un peu* (fine-tuning), et surtout sur de gros réseaux. Deuxième point : mettre des poids à zéro ne rend **pas** le calcul plus rapide par lui-même, car les zéros sont quand même multipliés. Pour gagner en vitesse il faut de l'élagage structuré ou du matériel qui exploite la parcimonie.
- **Quantification** : au lieu de stocker chaque poids sur 32 bits (nombre à virgule), on les représente sur 8 bits (entier), avec un facteur d'échelle. Le fichier est 2,2 fois plus petit (pas 4 fois, car le fichier contient aussi des échelles et des éléments non quantifiés), le calcul est plus rapide, et on perd ici environ 1 point d'exactitude. La **calibration** (la classe `Calibration` dans le script) fournit 200 signaux typiques pour que l'outil choisisse bien ses échelles. Pour du matériel embarqué (petit processeur, peu de mémoire), c'est une technique centrale.

Ce qu'on retient : **toujours mesurer le triplet précision / vitesse / taille**, jamais un seul des trois.

**Exercices.**

1. Réentraîne le modèle après élagage (quelques epochs). Récupère-t-on la performance ?
2. Mesure la latence avec des lots de 1, 16, 128 signaux. Qu'observes-tu ?
3. Cherche la différence entre quantification *statique* (utilisée ici) et *dynamique*, puis entre quantification *après* l'entraînement et *pendant* l'entraînement (QAT). Résume en cinq lignes.
4. Mesure la perte de rappel (et pas seulement d'exactitude) après quantification. Dans notre contexte, est-ce plus important ?

---

### Étape 9. Le réseau en C++ (`cpp/infer.cpp`)

**Objectif.** Faire tourner le réseau **sans aucune bibliothèque**, en C++ pur. C'est l'exercice qui répond à « Python ou C++ » dans l'offre, et qui t'oblige à comprendre ce que fait chaque couche.

**Lancer.**

```bash
cd cpp
python exporter_poids.py                    # écrit poids.txt et test_vecteurs.txt
g++ -O2 -std=c++17 -Wall -o infer infer.cpp # compile
./infer                                     # exécute
```

**Ce que tu dois voir.**

```
signal 0  PyTorch=...  C++=...  -> ...
...
signal 19  PyTorch=3.50754  C++=3.50754  -> DÉFAUT

Écart maximal PyTorch vs C++ : 2.86102e-06
```

L'écart est de trois millionièmes : ton code C++ fait **exactement** le même calcul que PyTorch.

**Comprendre.** Lis `infer.cpp` du bas vers le haut :

- `predire()` enchaîne `convolution`, `relu`, `maxpool2` quatre fois, puis la moyenne et la couche linéaire. C'est le schéma de la section 4, étape 4, écrit en clair.
- `convolution()` contient la formule : pour chaque filtre de sortie `o`, chaque position `t`, on somme `poids × entrée` sur tous les canaux d'entrée et les `k` positions du noyau, plus le biais. Le `if (s >= 0 && s < L)` correspond au remplissage par des zéros aux bords (`padding`).
- Les poids sont lus depuis un fichier texte, produit par `exporter_poids.py`. Les tableaux PyTorch sont rangés `(sortie, entrée, noyau)` ; le C++ les lit « à plat » avec l'indice `(o * c_in + i) * k + j`.

Ce que tu auras appris : le réseau, ce n'est que des multiplications et des additions rangées dans des boucles.

Une limite à connaître : ce code est **volontairement simple** et lent. Un vrai déploiement utiliserait une bibliothèque optimisée (ONNX Runtime a une interface C++, TensorRT, etc.), mais je ne l'ai pas testée ici, car elle demande de télécharger des binaires.

**Exercices.**

1. Ajoute un chronomètre (`<chrono>`) autour de `predire` et mesure le temps par signal. Compare à ONNX Runtime (étape 8). Compile aussi avec `-O0` puis `-O2` : quel facteur ?
2. Ajoute la sigmoïde et le seuil (0,24) pour que le programme affiche la probabilité et la décision.
3. Casse volontairement l'indexation (remplace `* c.k + j` par `* c.k`) et regarde l'écart avec PyTorch. Ce type de bug est silencieux : le programme ne plante pas, il se trompe.
4. Pour apprendre le C++ lui-même : fais les premiers chapitres de learncpp.com (variables, fonctions, `std::vector`, structures, boucles, flux de fichiers). Tu n'as pas besoin de plus pour lire ce programme.

---

### Étape 10. L'API (`08_api_flask.py`)

**Objectif.** Intégrer le modèle dans une application : un service qui reçoit un signal et répond « défaut » ou « saine ». C'est la tâche « intégrer dans l'application finale » de l'offre.

**Lancer le test intégré.**

```bash
python 08_api_flask.py --test
```

**Ce que tu dois voir.**

```
réel: défaut -> {'decision': 'DEFAUT', 'probabilite_defaut': 0.9934, 'seuil': 0.24}
réel: saine  -> {'decision': 'DEFAUT', 'probabilite_defaut': 0.3793, 'seuil': 0.24}
...
entrée invalide -> {'erreur': "'signal' doit être une liste de 512 nombres"}
```

Remarque que la deuxième ligne est une **fausse alarme** : 0,38 est au-dessus du seuil 0,24. C'est le prix du seuil choisi à l'étape 5, et c'est bien normal.

**Lancer le vrai serveur** (dans un terminal), puis l'appeler (dans un autre) :

```bash
python 08_api_flask.py
# autre terminal :
python -c "
from simulateur import generer; import json
X, y = generer(1, seed=5); print(json.dumps({'signal': X[0].tolist()}))" > signal.json
curl -s -X POST http://127.0.0.1:5000/diagnostic -H "Content-Type: application/json" -d @signal.json
```

**Comprendre.**

- Le service charge le modèle ONNX une seule fois au démarrage, et lit le seuil dans `sorties/seuil.json`.
- Le client envoie un signal **brut** ; l'API applique elle-même la même normalisation qu'à l'entraînement (`normaliser`). Oublier de reproduire exactement le même prétraitement en production est une des erreurs les plus fréquentes en ML.
- **Réflexe de cybersécurité** : l'entrée est validée avant d'être utilisée (type, longueur, valeurs finies et bornées). Une API qui accepte n'importe quoi est une surface d'attaque, même quand le modèle derrière est « juste » un classifieur.

**Exercices.**

1. Ajoute une route `GET /sante` qui renvoie `{"statut": "ok"}`.
2. Limite la taille des requêtes avec `app.config["MAX_CONTENT_LENGTH"]`. Teste avec un énorme JSON.
3. Journalise chaque requête (heure, décision, probabilité) dans un fichier, **sans** y écrire le signal complet. Pourquoi ce choix ?
4. Écris deux tests avec `pytest` (un cas valide, un cas invalide).
5. Quelle information l'API laisse-t-elle fuir à un attaquant qui l'interroge beaucoup (indice : la probabilité exacte) ? Réfléchis à l'extraction de modèle.

---

### Pour aller plus loin (facultatif, si le temps le permet)

- **Données réelles.** Cherche des jeux de données publics de contrôle non destructif (ultrasons ou courants de Foucault) sur Kaggle ou dans des articles scientifiques, et essaie ton modèle dessus. Je n'ai pas vérifié l'existence d'un jeu précis : vérifie la licence avant de t'en servir.
- **Déséquilibre des classes.** Génère un jeu avec 2 % de défauts (`taux_defaut=0.02`) et recommence l'évaluation : précision, courbe précision-rappel, pondération de la perte.
- **Comparer des architectures.** Réseau avec convolutions dilatées, ou un petit Transformer 1D. Pas parce que c'est à la mode : pour apprendre à comparer correctement.
- **Un A-scan par pixel : le C-scan.** Génère une image 2D en balayant la pièce, détecte les défauts sur l'image avec un CNN 2D.
- **Calibration des probabilités.** Quand le modèle dit 0,8, a-t-on raison 80 % du temps ? (courbe de fiabilité).

---

## 5. Calendrier : 10 semaines, du 26 octobre au 3 janvier

Le stage commence le 5 janvier 2027. Ce calendrier suppose 8 à 10 heures par semaine, avec **un creux volontaire pendant la période d'examens** de décembre. Ajuste les semaines 7 et 8 à ton vrai calendrier d'examens à Laval, que je ne connais pas.

L'ordre est pensé pour qu'à n'importe quel moment tu aies quelque chose à montrer si Evident t'appelle en entretien plus tôt que prévu.

| Semaine | Dates | Au programme | Livrable (commit Git) |
|---|---|---|---|
| **1** | 26 oct – 1 nov | Installation (section 3). Lire les sections 1 et 2. Vidéos 3Blue1Brown sur les réseaux de neurones. Commencer Karpathy (« micrograd »). **Étapes 0 et 1.** | Dépôt initialisé, journal commencé |
| **2** | 2 – 8 nov | Terminer micrograd, enchaîner avec « makemore » (partie 1). Calcul différentiel : dérivées, règle en chaîne, gradient (3Blue1Brown « Essence of calculus », ou Khan Academy). **Étapes 2 et 3.** | Baselines en place, premier graphique de signaux |
| **3** | 9 – 15 nov | **Étape 4** : entraîner le CNN, faire les exercices 1 à 4 (architecture, noyau, taux d'apprentissage, surapprentissage). | Courbes d'entraînement commentées dans le journal |
| **4** | 16 – 22 nov | **Étape 5** : évaluation, choix du seuil, intervalle de confiance par bootstrap, scénario « 1 % de défauts ». | Tableau de résultats avec incertitudes |
| **5** | 23 – 29 nov | **Étape 6** : robustesse, augmentation de données, comparaison avant/après, attaque adverse. | Graphique robustesse avant/après augmentation |
| **6** | 30 nov – 6 déc | **Étapes 7 et 8** : export ONNX, quantification, mesure précision/vitesse/taille. | Tableau d'optimisation |
| **7** | 7 – 13 déc | **Examens : 2 à 3 heures maximum.** Relire le journal. Commencer le README du dépôt. | README (brouillon) |
| **8** | 14 – 20 déc | **Examens : pause totale du projet.** | rien |
| **9** | 21 – 27 déc | **Étape 9** : C++ (learncpp.com puis `infer.cpp`, exercices 1 à 3). | Inférence C++ commentée |
| **10** | 28 déc – 3 jan | **Étape 10** : API. README final, GitHub public, mise à jour du CV, répétition des réponses d'entretien (section 6). | Dépôt public, CV mis à jour |

**Comment réagir si tu prends du retard.** L'ordre de priorité si tu dois couper : les étapes 4, 5 et 6 sont le cœur (entraîner, évaluer, tester la robustesse). L'étape 8 (optimisation), l'étape 9 (C++) et l'étape 10 (API) viennent ensuite. Mieux vaut un projet plus court mais **compris à fond** qu'un projet complet récité.

**Et ton projet RAG et ta feuille de route en sécurité des LLM ?** Je te suggère de les mettre en tâche de fond pendant ces dix semaines (une heure par semaine au plus, par exemple lancer un scan `garak`), puis d'y revenir en janvier. Cette décision t'appartient : elle dépend de ce que tu vises à long terme. Mais pour un stage d'IA industrielle, le projet de ce guide pèse plus.

---

## 6. Présenter le projet

### 6.1 Le dépôt GitHub

- Nom suggéré : `ndt-ultrasound-cnn`.
- Le `README.md` fourni explique l'installation, l'ordre d'exécution et les résultats. Complète-le avec **tes propres chiffres et tes images** (`exemples_signaux.png`, `evaluation.png`, `robustesse.png`).
- Ajoute une section **Limites** (reprends la section 10 de ce guide, avec tes mots). Un dépôt qui dit honnêtement ce qu'il ne prouve pas inspire plus confiance.
- Garde un historique de commits lisible : un commit par étape terminée.
- Ne publie pas de fichiers lourds ni de secrets (`.gitignore` déjà prévu).
- **Précise clairement que les données sont simulées.**

### 6.2 Sur le CV

Quand le projet est fini, remplace la section « projets » par quelque chose comme (à adapter à ce que tu as *vraiment* fait) :

> **Détection de défauts sur signaux ultrasonores simulés** (Python, PyTorch, ONNX, C++, Flask), janvier 2027
> - Conçu un CNN 1D (15 000 paramètres) comparé à une méthode à seuil et à une régression logistique ; évalué avec rappel, précision et AUC, seuil choisi pour un rappel de 95 %
> - Testé la robustesse (bruit, étalonnage, attaque adverse FGSM) et mesuré l'effet de l'augmentation de données
> - Quantifié en INT8 (modèle 2,2 fois plus petit, environ 1,8 fois plus rapide, environ 1 point d'exactitude perdu) ; réimplémenté l'inférence en C++ pur, vérifiée contre PyTorch (écart de 3·10⁻⁶) ; exposé via une API validée

N'écris que les chiffres que **tu** as mesurés sur ta machine.

### 6.3 La lettre

La lettre actuelle dit que tu « prévois » d'évaluer avec garak et promptfoo, et que ton plan est hebdomadaire. Si tu es convoqué en entretien, prépare une version à jour : ce sont les résultats de ce projet que tu citeras.

### 6.4 Préparer l'entretien

**Le pitch en une minute** (à adapter et à dire à voix haute plusieurs fois) :

> « Je viens de la cybersécurité et je n'avais qu'un cours d'introduction à l'IA. J'ai voulu comprendre les fondements, alors j'ai monté un projet complet : un CNN 1D qui détecte des défauts sur des signaux ultrasonores que j'ai simulés. J'ai commencé par des méthodes simples pour savoir quoi battre, puis entraîné le réseau, évalué avec le rappel plutôt que l'exactitude parce qu'un défaut raté coûte plus cher qu'une fausse alarme, testé sa robustesse, quantifié le modèle, et réimplémenté l'inférence en C++ pour vérifier que je comprenais chaque opération. Les données sont simulées, donc je ne prétends pas connaître les vrais signaux, mais je connais la méthode et je veux l'appliquer à vos données. »

**Questions probables et ce qu'elles testent.**

| Question | Ce qu'on vérifie | Ta ligne de réponse |
|---|---|---|
| « Pourquoi un réseau et pas un seuil ? » | Esprit critique | « J'ai mesuré : le seuil donne 65 %, le réseau 86 %. Et si le seuil avait suffi, je l'aurais gardé. » |
| « Comment sais-tu que ton modèle est bon ? » | Rigueur d'évaluation | Trois jeux séparés, métriques adaptées, incertitude sur les résultats, test jamais utilisé pour régler quoi que ce soit |
| « Pourquoi ce seuil de 0,24 ? » | Compréhension du compromis | Rappel de 95 % visé, choisi sur la validation, coût : 43 % de fausses alarmes, et dire comment ça change si les défauts sont rares |
| « Que se passe-t-il si les défauts sont rares ? » | Maturité | Le calcul de la section 4, étape 5 : précision d'environ 2 % à 1 % de prévalence |
| « Qu'est-ce qui limite ton modèle ? » | Honnêteté | Données simulées, un seul type de défaut, classes équilibrées, fragile aux perturbations adverses |
| « Comment le déploierais-tu sur un appareil ? » | Ingénierie | Export ONNX, quantification INT8, vérification d'équivalence, mesure de latence sur la cible |
| « Tu n'as jamais entraîné de réseau en entreprise. » | Gestion de la lacune | « C'est vrai. J'ai entraîné, évalué, optimisé et déployé un modèle de bout en bout sur un projet à ma portée, et je sais apprendre vite : voici le plan que j'ai suivi. » |
| « Que sais-tu de ce qu'on fait ? » | Intérêt réel | **Lis le site d'Evident avant l'entretien** : produits, types d'inspection, publications. N'invente rien. |

---

## 7. Ressources pour apprendre

Je n'ai pas pu vérifier en direct que ces liens existent encore aujourd'hui : cherche les titres exacts.

**Les réseaux de neurones, du plus doux au plus technique**

- **3Blue1Brown, série « Neural networks »** (YouTube, 4 vidéos, environ 1 heure). L'intuition visuelle, à voir en premier.
- **StatQuest (Josh Starmer), playlist « Neural Networks »** (YouTube). Très patient, idéal pour un débutant.
- **Andrej Karpathy, « Neural Networks: Zero to Hero »** (YouTube, gratuit). Tu codes la rétropropagation à la main (« micrograd »), puis des modèles de plus en plus gros. Long, mais la meilleure compréhension en profondeur. Commence par la première vidéo.
- **Tutoriels PyTorch « Learn the Basics »** (pytorch.org/tutorials).
- **« Dive into Deep Learning »** (d2l.ai), livre gratuit avec code PyTorch, pour approfondir les CNN.
- **fast.ai, « Practical Deep Learning for Coders »**, cours gratuit, très orienté pratique.
- **Andrew Ng, « Machine Learning Specialization »** (Coursera). Vérifie les conditions d'accès gratuit actuelles.

**Maths**

- **3Blue1Brown, « Essence of calculus »** (dérivées, règle en chaîne) et **« Essence of linear algebra »** pour compléter ton cours de maths.
- Ton fichier `maths_pour_IA_RAG.md` couvre déjà l'algèbre linéaire et les métriques. Le calcul différentiel (dérivées) est le complément à ajouter.

**Évaluation**

- La documentation de scikit-learn sur les métriques de classification (`precision_recall_fscore_support`, courbes ROC et précision-rappel).

**Contrôle non destructif**

- Cherche « Introduction to NDT » et « ultrasonic testing basics A-scan » ; le **NDT Resource Center** (nde-ed.org) propose des cours d'introduction en ligne. Cherche aussi « eddy current testing basics ».
- Lis les pages produits et les études de cas du site d'Evident.

**C++**

- **learncpp.com**, cours gratuit et complet. Pour ce projet, les premiers chapitres suffisent : variables, fonctions, `std::vector`, structures, boucles, fichiers.

---

## 8. Glossaire

| Terme | Sens |
|---|---|
| **A-scan** | Signal ultrasonore enregistré en un point : amplitude en fonction du temps |
| **Activation (fonction d')** | Fonction non linéaire appliquée à la sortie d'un neurone (ReLU, sigmoïde...) |
| **Adam** | Variante de la descente de gradient qui adapte le pas à chaque poids |
| **AUC** | Aire sous la courbe ROC : qualité du classement des signaux, de 0,5 (hasard) à 1 (parfait) |
| **Attaque adverse** | Perturbation minuscule, calculée pour tromper un modèle (FGSM en est une) |
| **Augmentation de données** | Ajouter des variations artificielles (bruit, gain, décalage) aux exemples d'entraînement |
| **Batch / mini-lot** | Petit paquet d'exemples traités ensemble à chaque mise à jour |
| **Biais (b)** | Constante ajoutée à la somme pondérée d'un neurone |
| **Calibration** | (quantification) choisir les échelles à partir d'exemples typiques ; (probabilités) vérifier que 0,8 signifie bien 80 % |
| **CND / NDT** | Contrôle non destructif : inspecter sans abîmer |
| **CNN** | Réseau de neurones convolutif |
| **Convolution** | Un filtre glisse le long du signal et calcule une somme pondérée à chaque position |
| **Descente de gradient** | Ajuster les poids dans la direction qui réduit la perte |
| **Epoch** | Un passage complet sur le jeu d'entraînement |
| **Entropie croisée** | Fonction de perte pour la classification |
| **Étalonnage** | Réglage de l'appareil sur une référence connue |
| **Gradient** | Pour chaque poids, de combien la perte varie si on le modifie un peu |
| **Inférence** | Utiliser un modèle entraîné pour prédire |
| **INT8** | Entiers sur 8 bits, utilisés pour compacter et accélérer un modèle |
| **Logit** | Sortie brute du réseau avant la sigmoïde |
| **Matrice de confusion** | Tableau des vrais/faux positifs/négatifs |
| **ONNX** | Format de fichier d'échange de modèles, indépendant du framework |
| **Paramètres** | Les poids et biais du modèle |
| **Perte (loss)** | Nombre mesurant l'erreur du modèle ; l'entraînement le fait baisser |
| **Poids (w)** | Nombres appris qui définissent le modèle |
| **Pooling** | Réduit la longueur en gardant le maximum (ou la moyenne) d'un groupe |
| **Précision** | Parmi les alarmes, part de vraies détections |
| **Quantification** | Représenter les poids sur moins de bits |
| **Rappel (recall)** | Parmi les vrais défauts, part détectée |
| **ReLU** | `max(0, z)` |
| **Rétropropagation** | Calcul de tous les gradients, de la sortie vers l'entrée |
| **ROC (courbe)** | Rappel en fonction du taux de fausses alarmes, pour tous les seuils |
| **Seuil** | Probabilité à partir de laquelle on déclenche l'alarme |
| **Sigmoïde** | Fonction qui transforme un logit en probabilité entre 0 et 1 |
| **Surapprentissage** | Le modèle apprend le bruit du jeu d'entraînement et généralise mal |
| **Taux d'apprentissage (lr)** | Taille du pas dans la descente de gradient |
| **Zone morte** | Début du signal inexploitable à cause de l'écho d'entrée |

---

## 9. Dépannage

| Symptôme | Cause probable et remède |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | L'environnement virtuel n'est pas activé : `source .venv/bin/activate` |
| `FileNotFoundError: sorties/cnn_meilleur.pt` | Les scripts se suivent. Lance `03_train_cnn.py` avant `04`, `05`, `06` |
| Perte bloquée à environ 0,693 et exactitude à 0,50 | Le modèle n'apprend pas. Vérifie le taux d'apprentissage (essaie 1e-3), que `normaliser` est bien appliqué, que `y` est de type `float` pour la perte |
| Mes chiffres diffèrent un peu de ceux du guide | Normal (versions, arrondis). Un écart de quelques points est acceptable ; un score de 0,50 ne l'est pas |
| Entraînement très lent | Réduis le nombre d'epochs (`python 03_train_cnn.py 30`) ou la taille du jeu |
| `torch.onnx.export` échoue après une mise à jour | Retire `dynamo=False` et fais `pip install onnxscript` |
| Erreur dans `07_quantification.py` | `pip install -U onnx onnxruntime`. Vérifie que `sorties/cnn.onnx` existe (étape 7) |
| `g++: command not found` | `xcode-select --install` |
| `Address already in use` au lancement de l'API | Sur les Mac récents, le port 5000 est parfois pris par « Récepteur AirPlay ». Remplace `port=5000` par `5001` dans le script, et dans la commande `curl` |
| Le dépôt Git est énorme | Vérifie `.gitignore` : `.venv/` et les fichiers `.pt` / `.onnx` ne doivent pas être commités |

Si tu es bloqué plus d'une heure, écris le message d'erreur dans ton journal, **décris ce que tu as essayé**, puis cherche l'erreur exacte sur Internet. Savoir chercher une erreur fait aussi partie du métier.

---

## 10. Ce que ce projet ne prouve pas

À garder en tête, et à dire spontanément en entretien :

- **Les données sont simulées.** Elles reproduisent la forme générale d'un A-scan, pas la physique réelle (diffusion, modes d'onde, géométries complexes). Un modèle qui réussit sur un simulateur peut échouer sur de vraies données : c'est même le cas typique.
- **Un seul type de défaut**, un seul écho, une seule sonde. Le vrai travail traite de nombreux types de défauts, de matériaux et d'appareils.
- **Les classes sont équilibrées** (50/50). En réalité, les défauts sont rares et c'est ce qui rend l'évaluation difficile (voir l'étape 5).
- **Aucun expert n'a étiqueté les signaux.** En vrai, obtenir des étiquettes fiables est souvent le plus dur et le plus coûteux.
- **Le déploiement est démonstratif.** Une vraie intégration dans un appareil industriel implique des contraintes de mémoire, de temps réel et de validation que ce projet ne touche pas.
- **Tu n'es pas devenu spécialiste du contrôle non destructif.** Tu as appris la **méthode** : formuler, comparer à une référence, entraîner, évaluer honnêtement, tester la robustesse, optimiser, déployer. C'est exactement ce qui se transfère, et c'est ce que le stage va te permettre d'appliquer à de vraies données.

Bon travail. Tu as dix semaines. Commence par la section 3, puis l'étape 0.
