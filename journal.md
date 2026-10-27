# Journal d'apprentissage : réseaux de neurones et projet CNN

Je tiens ce journal pour garder une trace de ce que j'apprends au fil du projet, avec mes questions et les réponses que j'ai obtenues. Les réponses aux exercices du guide sont rangées à part, dans `projet_ndt_cnn/reponses_exercices.md`.

## Lundi 26 octobre 2026 (semaine 1)

### Installation de l'environnement

J'ai créé l'environnement virtuel du projet avec `python3 -m venv .venv`, puis lancé `pip install -r requirements.txt`. Ça a échoué tout de suite :

```
ERROR: Could not find a version that satisfies the requirement numpy (from versions: none)
```

J'ai d'abord cru que numpy n'existait pas pour ma version de Python. En fait, `from versions: none` veut dire que pip n'a trouvé aucun fichier du tout, même pas le code source, ce qui pointe vers un problème de connexion à PyPI.

Pour chercher la cause, j'ai vérifié trois choses :
- `pip config list` ne renvoyait rien, donc pas de faux index configuré ;
- `env | grep -i proxy` ne renvoyait rien, donc pas de proxy ;
- `curl -I https://pypi.org/simple/numpy/` répondait `HTTP/2 200`, donc mon Mac joignait bien PyPI.

C'est `pip install -v numpy` qui a donné la vraie raison : un `ReadTimeoutError`. Ma connexion est trop lente pour le délai de 15 secondes que pip accorde par défaut, et la page d'index de numpy pèse à elle seule environ 2,3 Mo. La solution est d'allonger le délai :

```bash
pip config set global.timeout 120
pip config set global.retries 10
```

J'ai aussi noté que ma venv tourne sous Python 3.14.6 (Homebrew). On m'a conseillé de passer à Python 3.12, parce que `torch` et `onnxruntime` arrivent souvent en retard sur les nouvelles versions de Python.

Ce que je retiens : un message d'erreur ne dit pas toujours la vraie cause. Le mode verbeux (`-v`) et quelques tests simples m'ont permis de l'isoler.

### Où sont les neurones dans `00_backprop_numpy.py`

Je cherchais des « neurones » dans le code et je n'en trouvais pas. Il n'existe aucun objet neurone : un neurone, c'est une colonne de la matrice de poids, une case du vecteur de biais et une fonction d'activation.

```python
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros((1, 4))   # couche cachée
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros((1, 1))   # couche de sortie
```

`W1` a 2 lignes (les 2 entrées) et 4 colonnes (les 4 neurones cachés). `W2` n'a qu'une colonne, donc un seul neurone de sortie. Le réseau a 5 neurones en tout. Les 2 entrées n'en sont pas, ce sont les données.

Les neurones « travaillent » dans la fonction `avant`, sur la ligne `h = np.tanh(X @ W1 + b1)`. Elle calcule la sortie des 4 neurones pour les 4 exemples à la fois. Ça revient exactement à une double boucle sur les exemples et les neurones, en plus rapide.

### Lire la ligne `W1, b1 = ...`

Cette ligne fait deux affectations d'un coup (déballage de tuple) : `W1` reçoit la première valeur, `b1` la seconde.

- `rng.normal(0, 1, (2, 4))` tire 8 nombres au hasard selon une loi normale de moyenne 0 et d'écart-type 1, rangés en 2 lignes et 4 colonnes. `rng` est le générateur créé avec la graine 1, ce qui donne les mêmes nombres à chaque exécution.
- `np.zeros((1, 4))` crée une ligne de 4 zéros. La parenthèse intérieure est la forme, passée comme un seul argument.

J'ai compris pourquoi les poids sont aléatoires alors que les biais partent de zéro. Si tous les poids valaient 0, les 4 neurones calculeraient la même chose et recevraient les mêmes corrections : ils resteraient identiques et je n'aurais en pratique qu'un seul neurone. Le hasard sert à les différencier dès le départ.

### Compter les neurones en lisant du code

La règle que j'ai retenue : une matrice de poids a la forme (entrées, neurones), et le second nombre donne le nombre de neurones de la couche. Je peux vérifier de deux façons. Le biais a autant de cases qu'il y a de neurones, et le nombre de neurones d'une couche est égal au nombre d'entrées de la suivante.

La même règle marche avec PyTorch :
- dans `01_premier_reseau.py`, `nn.Linear(64, 32)` puis `nn.Linear(32, 10)` font 32 + 10 = 42 neurones ;
- dans `modele.py`, `nn.Conv1d(entrées, filtres, noyau)` donne 8, 16, 32 et 32 filtres, puis `nn.Linear(32, 1)` ajoute le neurone de sortie. Dans un CNN, on compte des filtres : un filtre est un neurone dont on fait glisser les poids le long du signal.

Les activations (`tanh`, `ReLU`, sigmoïde) et le pooling n'ont pas de poids, donc ils ne comptent pas comme des neurones. `noyau=9` non plus : c'est le nombre de points du signal qu'un filtre regarde à la fois.

### Le gradient et la rétropropagation

L'image qui m'a aidé : je suis dans la montagne, dans le brouillard, et je veux descendre. L'altitude, c'est la perte (l'erreur du réseau). Ma position, ce sont les poids. La pente sous mes pieds, c'est le gradient.

Le gradient d'un poids répond à la question « si j'augmente un peu ce poids, de combien la perte change-t-elle ? ». S'il est positif, il faut diminuer le poids, et inversement. D'où la ligne d'entraînement `W1 -= LR * dW1`, où `LR = 0.5` est la longueur du pas. `dW1` a toujours la même forme que `W1`, avec un gradient par poids.

La rétropropagation part de l'erreur en sortie et la fait remonter couche par couche, en appliquant la règle de la chaîne. Dans la fonction `gradients` :
- `dz2 = (p - y) / len(X)` est l'erreur en sortie (prédiction moins réponse attendue, divisée par 4 parce que la perte est une moyenne) ;
- `dW2 = h.T @ dz2` dit qu'un poids est responsable de l'erreur en proportion de ce que son neurone envoyait ;
- `dh = dz2 @ W2.T` renvoie l'erreur vers chaque neurone caché, selon le poids qui le relie à la sortie ;
- `dz1 = dh * (1 - h ** 2)` traverse la fonction tanh, dont la dérivée est `1 - h²`. Un neurone saturé (h proche de 1 ou de -1) laisse presque rien passer et apprend lentement ;
- `dW1 = X.T @ dz1` refait le même calcul une couche plus tôt.

Pour une couche, il y a deux calculs à faire, peu importe la profondeur du réseau : gradient des poids = (entrée de la couche).T @ (erreur de la couche), puis erreur de la couche précédente = (erreur) @ (poids).T × (dérivée de l'activation).

Le bloc de vérification du code calcule aussi le gradient « à la main » : il fait bouger un poids de ±0,000001 et mesure la variation de la perte. Si cette valeur colle à celle de la rétropropagation, les formules sont justes. Cette méthode est trop lente pour un vrai réseau, car il faudrait deux passages avant par poids. La rétropropagation obtient tous les gradients en un seul passage arrière.

### L'opérateur `@`

`@` est le produit matriciel (`np.matmul`). Ce n'est pas la même chose que `*`, qui multiplie case par case. Chaque case de `A @ B` se calcule en prenant une ligne de A et une colonne de B, en multipliant les éléments deux à deux puis en additionnant. C'est exactement la somme pondérée d'un neurone.

`A @ B` ne marche que si le nombre de colonnes de A est égal au nombre de lignes de B : `(4, 2) @ (2, 4)` donne `(4, 4)`. C'est pour ça qu'il y a des `.T` (transposées) dans les gradients. En cas d'erreur de dimension, je dois afficher les `.shape` des deux matrices.

### Exercice : passer la couche cachée de 4 à 8 neurones

J'ai modifié les deux lignes d'initialisation :

```python
W1, b1 = rng.normal(0, 1, (2, 8)), np.zeros((1, 8))
W2, b2 = rng.normal(0, 1, (8, 1)), np.zeros((1, 1))
```

C'était correct, et je n'avais rien à toucher dans `avant` ni dans `gradients`, parce que `@`, `.T` et `.sum` s'adaptent aux formes. J'avais oublié de mettre à jour le commentaire en haut du fichier, qui parlait encore de 4 neurones.

Avec 8 neurones, le réseau apprend XOR : la perte tombe à 0,0013 et les prédictions sont `[0.001 0.999 0.999 0.002]`.

J'ai d'abord pensé que 8 neurones « corrigeaient » la graine 0, celle qui bloque le réseau à 4 neurones. Le test sur 200 graines montre autre chose :

| Neurones cachés | Échecs sur 200 graines |
|---|---|
| 2 | 101 (environ 50 %) |
| 4 | 5 (2,5 %), dont la graine 0 |
| 8 | 0 |
| 16 | 0 |

La graine 0 n'avait rien de spécial, elle faisait partie des départs malchanceux. Plus de neurones donne au réseau plus de chemins vers une solution, donc moins de risques de rester coincé dans un minimum local. Ce n'est pas une garantie : d'autres réglages (taux d'apprentissage, nombre d'étapes) peuvent encore faire échouer un départ.

Ma conclusion pour l'exercice : passer de 4 à 8 neurones rend l'apprentissage plus robuste à l'initialisation aléatoire, et sur 200 graines le taux d'échec passe de 2,5 % à 0 %.

### Micrograd et la différentiation automatique

Mon planning mentionnait « Karpathy (micrograd) » et je ne savais pas ce que c'était. Micrograd est un petit programme Python (environ 150 lignes) qu'Andrej Karpathy construit pas à pas dans la vidéo « The spelled-out intro to neural networks and backpropagation: building micrograd », la première de sa série « Neural Networks: Zero to Hero ».

Il calcule les gradients automatiquement. Chaque nombre retient les opérations qui l'ont produit, et un appel à `.backward()` remonte ce chemin avec la règle de la chaîne. Dans `00_backprop_numpy.py`, j'ai écrit les gradients à la main, et ça ne marche que pour ce réseau-là. Micrograd fait la même chose pour n'importe quel calcul.

C'est le même mécanisme que `l.backward()` dans PyTorch, en version minuscule, et c'est aussi le sujet du module `chain-rule-and-autodiff` du cours AI Engineering from Scratch. Je vois la progression comme ça : gradients à la main (fichier 00), puis automatiques avec un moteur que j'écris moi-même (micrograd), puis automatiques avec PyTorch (fichier 01).

Le conseil retenu : regarder la vidéo avec un notebook ouvert et retaper le code en même temps.

Pour suivre la vidéo, j'ai préparé un dossier `micrograd/` à part, à côté de `projet_ndt_cnn`, puisque c'est un exercice d'apprentissage et pas une partie du projet. Il a sa propre venv, avec `jupyter`, `graphviz`, `matplotlib` et `numpy`. J'ai aussi installé graphviz avec `brew` : la fonction `draw_dot` de Karpathy, qui dessine les graphes de calcul, a besoin à la fois du paquet Python et du programme de dessin. Cette fois, l'installation est passée sans erreur, grâce au timeout allongé. Je code dans un notebook Jupyter dans PyCharm, avec l'interpréteur réglé sur `micrograd/.venv`.

### À quoi sert Jupyter Notebook

Un notebook découpe le code en cellules que j'exécute une par une avec Shift + Entrée. Le résultat s'affiche juste en dessous, qu'il s'agisse d'un nombre, d'un tableau ou d'un graphique.

Ce qui change par rapport à un `.py`, c'est que les variables restent en mémoire entre les cellules. Un programme Python invisible, le noyau (kernel), tourne en arrière-plan et garde tout ce que j'ai créé. Je peux donc définir un réseau dans une cellule, l'entraîner dans une autre, puis modifier une seule ligne et relancer seulement cette cellule, sans tout réexécuter depuis le début. Avec un `.py`, chaque exécution repart de zéro.

Pour apprendre, ça me permet de tester une idée, regarder le résultat, changer un détail et recommencer en quelques secondes. Les graphes de `draw_dot` s'affichent directement dans la page, et je peux ajouter des cellules de texte (Markdown) pour prendre des notes à côté du code. C'est pour ça que Karpathy l'utilise dans ses vidéos.

Le piège : le noyau se souvient de l'ordre dans lequel j'ai exécuté les cellules, pas de leur ordre dans la page. Si j'exécute les cellules dans le désordre ou que j'en supprime une, une variable peut rester en mémoire alors que le code qui la créait a disparu. Le remède est de redémarrer le noyau et de tout relancer dans l'ordre (Restart & Run All). Si le notebook marche après ça, il est fiable.

Le notebook sert à explorer et à apprendre. Le code définitif du projet, comme `modele.py` ou `03_train_cnn.py`, reste dans des fichiers `.py`, plus faciles à relancer, à importer et à suivre avec Git.

### Vidéos 3Blue1Brown

Dans la série « Neural networks », je m'arrête après la vidéo 4, « Backpropagation calculus ». Les quatre premières font environ une heure, et chacune correspond à une partie de mon fichier 00 : les neurones et les poids, la descente de gradient, la rétropropagation intuitive, puis le calcul des dérivées. Les vidéos suivantes portent sur les LLM et les Transformers, qui ne sont pas au programme du projet.

J'ai regardé les quatre vidéos. J'ai compris le principe de la rétropropagation et les mathématiques qui le sous-tendent : les dérivées, la règle de la chaîne et le gradient.

## Mardi 27 octobre 2026

### Le modèle et l'optimiseur dans `01_premier_reseau.py`

```python
modele = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
opt = torch.optim.Adam(modele.parameters(), lr=1e-2)
```

La première ligne construit le réseau. `nn.Sequential` enchaîne les couches dans l'ordre : l'entrée traverse la première, puis la deuxième, puis la troisième. `nn.Linear(64, 32)` est une couche de 32 neurones qui reçoit les 64 pixels d'une image. Elle contient ses poids et ses biais, déjà initialisés au hasard par PyTorch : c'est l'équivalent de mon `W1, b1 = ...` du fichier 00, en une seule instruction. `nn.ReLU()` est la fonction d'activation, qui garde les valeurs positives et remplace les négatives par 0. Elle joue le rôle de `np.tanh`. `nn.Linear(32, 10)` est la couche de sortie, avec un neurone par chiffre de 0 à 9.

Écrire `modele(X)` fait le passage avant, comme ma fonction `avant`.

Un détail qui peut piéger : PyTorch range les poids de `nn.Linear(64, 32)` dans une matrice de forme (32, 64), c'est-à-dire (sorties, entrées), à l'inverse de mon fichier NumPy. Ça ne change rien au calcul, PyTorch transpose tout seul.

La deuxième ligne crée l'optimiseur, l'objet qui met les poids à jour. C'est lui qui remplace ma ligne `W1 -= LR * dW1; b1 -= LR * db1; ...`. `modele.parameters()` lui donne la liste de tout ce qu'il doit modifier : les poids et biais des deux couches linéaires, soit 2 410 nombres. `lr=1e-2` est le taux d'apprentissage, 0,01, l'équivalent de mon `LR`.

Adam ne se contente pas de soustraire `lr × gradient`. Il garde en mémoire une moyenne des gradients précédents, ce qui donne de l'élan dans les directions qui reviennent souvent, et il adapte la taille du pas pour chaque poids. En général, il converge plus vite et demande moins de réglages que la descente de gradient simple. L'équivalent exact de mon fichier 00 serait `torch.optim.SGD`.

Dans la boucle d'entraînement, trois lignes font tout le travail de mon fichier 00 :

| PyTorch | Ce que ça fait | Équivalent dans le fichier 00 |
|---|---|---|
| `opt.zero_grad()` | remet les gradients à zéro | rien : ma fonction recalculait des gradients neufs à chaque appel |
| `l.backward()` | calcule tous les gradients | toute la fonction `gradients` |
| `opt.step()` | met les poids à jour | `W1 -= LR * dW1` et les trois autres |

`zero_grad` est nécessaire parce que PyTorch additionne les nouveaux gradients aux anciens au lieu de les remplacer. Sans cette remise à zéro, chaque étape utiliserait la somme de tous les gradients passés.

### Ce que mesure vraiment un gradient

Je pensais que le gradient indiquait, pour chaque neurone, comment la valeur du neurone précédent l'affecte. Ce n'est pas tout à fait ça. Un gradient mesure toujours l'effet d'un nombre sur la perte, à la fin du réseau. `dW1[0, 2]` répond à la question : si j'augmente un peu ce poids, de combien l'erreur finale change-t-elle ?

Ce que je décrivais, l'effet d'un neurone sur le neurone suivant, est une dérivée locale. Pour une couche linéaire, elle vaut simplement le poids qui les relie : si le neurone caché k augmente de 1, la valeur de sortie avant activation augmente de `W2[k]`. Ce n'est qu'un maillon de la chaîne.

La rétropropagation multiplie ces maillons locaux le long du chemin jusqu'à la perte, c'est la règle de la chaîne. On le voit dans le code : `dh = dz2 @ W2.T` multiplie l'erreur en sortie par le poids qui relie chaque neurone caché à la sortie (l'effet local), puis `* (1 - h ** 2)` ajoute le maillon de tanh. Le résultat dit comment chaque neurone caché affecte la perte, et pas seulement le neurone suivant.

Ce qu'on corrige à la fin, ce sont les poids et les biais, pas les valeurs des neurones : elles dépendent de l'entrée et changent à chaque exemple. Les gradients par rapport aux neurones (`dh`, `dz1`) sont des étapes intermédiaires qui servent à calculer ceux des poids (`dW1`).

En résumé, un gradient dit de combien la perte change quand on bouge un poids, et dans quel sens. C'est un taux : un gradient de 0,3 veut dire que la perte augmente d'environ 0,3 fois le petit changement appliqué au poids. Cette estimation n'est fiable que pour de petits pas, parce que la pente change dès qu'on se déplace. C'est pour ça qu'on avance par petits pas et qu'on recalcule les gradients à chaque étape.

### Exercices sur `01_premier_reseau.py`

Exercice 1, 100 epochs. L'exactitude sur le test reste à 0,969 entre les epochs 70 et 90, puis vaut 0,971 aux epochs 95 et 100. Elle se stabilise autour de 97 %. Le jeu de test contient 450 images, donc une image correspond à environ 0,0022 d'exactitude : passer de 0,969 à 0,971, c'est une seule image de plus bien classée.

Exercice 2, taille de la couche cachée. Avec 128 neurones, j'ai fini à 0,971, avec un pic à 0,976 pendant l'entraînement. L'écart entre le pic et la fin représente deux images, donc je ne dois pas trop l'interpréter. Il me reste à faire l'essai avec 8 neurones et à comparer les valeurs finales des trois tailles (8, 32, 128).

Exercice 3, sans couche cachée. Ma première version était fausse : j'avais écrit `nn.Sequential(nn.Linear(64, 10), nn.ReLU(), nn.Linear(10, 10))`, qui garde une couche cachée de 10 neurones. Mon 0,958 mesure donc un réseau avec une couche cachée étroite, plus proche de l'exercice 2 que de l'exercice 3. Le modèle demandé est `nn.Linear(64, 10)` seul. Je dois refaire l'essai avant de conclure sur l'utilité des couches cachées.

En refaisant l'essai, le modèle sans couche cachée finit avec une perte de 0,324, contre 0,011 avec la couche cachée. Mais cette perte est calculée sur les données d'entraînement (`perte(modele(Xtr), ytr)`), donc elle dit à quel point le modèle colle aux images qu'il a déjà vues, pas s'il prédit bien des images nouvelles. Un modèle avec plus de paramètres (2 410 contre 650) peut toujours mieux coller à l'entraînement, y compris en apprenant par cœur. Pour juger la prédiction, il faut comparer l'exactitude sur le test, que le programme affiche à côté de la perte.

Résultat sur le test : 0,936 sans couche cachée, 0,971 avec une couche cachée de 128 neurones. Sur 450 images, ça fait environ 29 erreurs contre 13 : la couche cachée divise les erreurs par plus de deux. Elle améliore donc vraiment la prédiction, pas seulement l'ajustement à l'entraînement.

Deux nuances. Le modèle linéaire atteint déjà 93,6 % : ce problème est en grande partie faisable avec une simple somme pondérée des pixels. La couche cachée apporte les 3 à 4 points restants, en combinant des pixels entre eux, ce qu'un modèle linéaire ne sait pas faire. Et l'écart de perte (0,324 contre 0,011, presque 30 fois moins) est bien plus grand que l'écart d'exactitude : une partie du gain mesuré à l'entraînement ne se retrouve pas sur des images nouvelles. C'est pour ça qu'on juge un modèle sur le test.

Exercice 4, sans `opt.zero_grad()`. L'exactitude tombe à 0,536. PyTorch additionne chaque nouveau gradient dans `.grad` au lieu de le remplacer. Sans remise à zéro, le gradient utilisé à l'étape t est la somme de tous les gradients depuis le début, et la plupart ont été calculés à des endroits où les poids ne sont plus. Le pas suit donc surtout l'ancienne pente au lieu de la pente actuelle.

Je l'ai compris sur un exemple à une dimension : la perte w², avec w = 1 au départ et un pas de 0,1. Avec accumulation, w passe de 1 à 0,8 puis 0,44, puis presque 0 à l'étape 3, c'est-à-dire le minimum. Mais la somme des gradients vaut encore 4,48, parce qu'elle contient toutes les anciennes pentes positives. Le pas suivant envoie w à -0,45, puis -0,81, puis -1. Le réseau dépasse la vallée et oscille au lieu de s'arrêter au fond, comme une bille qui n'oublie jamais son élan.

PyTorch accumule exprès, pour deux raisons. Quand un poids sert à plusieurs endroits du calcul, son gradient total est la somme de ses contributions, ce que je devrais retrouver dans le `backward` de micrograd. Et quand un lot de données est trop gros pour la mémoire, on peut le couper en morceaux, appeler `backward()` sur chacun et n'appeler `step()` qu'une fois. Le prix à payer, c'est de remettre à zéro soi-même à chaque étape.

### Repérer un défaut à l'œil dans un signal bruité

Avec `difficulte=2.0`, j'ai cru voir deux défauts sur un des signaux défectueux. En vérifiant ce que le simulateur avait vraiment placé, je me suis trompé : un signal défectueux ne contient qu'un seul défaut, et les pics que je voyais étaient du bruit ou l'écho de fond. Le vrai défaut avait une amplitude deux fois plus petite que l'écart-type du bruit.

Ce qui m'a surpris, c'est la facilité avec laquelle l'œil trouve des motifs dans du bruit, et avec assurance. Un pic qui dépasse un peu les autres ressemble à un défaut, même quand il n'y en a pas. Pour juger un détecteur, humain ou réseau, il faut donc comparer à la vérité connue, pas se fier à l'impression. Et je comprends mieux l'intérêt du CNN : ses filtres peuvent reconnaître la forme d'une impulsion (une petite oscillation précise) là où je ne regarde que la hauteur des pics.

![Le signal DÉFAUT avec et sans bruit : le défaut est vers l'échantillon 351](projet_ndt_cnn/sorties/defaut_annote.png)

### Questions encore ouvertes

- Vérifier que l'installation passe avec le timeout allongé, et décider si je recrée la venv sous Python 3.12.
- Commencer la vidéo micrograd de Karpathy, maintenant que les vidéos 3Blue1Brown sont terminées.
