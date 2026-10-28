# Journal d'apprentissage : réseaux de neurones et projet CNN

Je tiens ce journal pour garder une trace de ce que j'apprends au fil du projet, avec mes questions et les réponses que j'ai obtenues. Les réponses aux exercices du guide sont rangées à part, dans `reponses_exercices.md`.

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

![Le signal DÉFAUT avec et sans bruit : le défaut est vers l'échantillon 351](sorties/defaut_annote.png)

### Ce qu'est un écho de fond

Le capteur ultrasonore envoie une courte impulsion dans la pièce, puis écoute ce qui revient. Une partie du son rebondit tout de suite sur la surface d'entrée : c'est l'écho d'entrée, le grand pic vers l'échantillon 50. Le reste traverse le métal jusqu'à la face opposée, rebondit et revient : c'est l'écho de fond. Il arrive plus tard parce que le son a fait l'aller-retour dans toute l'épaisseur, et il est plus faible parce que le matériau a absorbé une partie de l'énergie en chemin.

Sur un A-scan, le temps correspond donc à la profondeur. Plus la pièce est épaisse, plus l'écho de fond arrive tard, et c'est pour ça qu'il se déplace autant d'un signal à l'autre dans la figure des 20 signaux superposés.

Un défaut (une fissure, une bulle) est un obstacle entre les deux faces. Il renvoie un petit écho entre l'écho d'entrée et l'écho de fond, et il bloque une partie du son, ce qui rend l'écho de fond un peu plus faible : le simulateur appelle ça l'ombre du défaut (ligne 62). La position de l'écho du défaut indique à quelle profondeur il se trouve.

### La simplification du simulateur la plus éloignée de la réalité

Pour moi, c'est la façon dont le simulateur représente le défaut, l'obstacle dans la pièce (lignes 61 à 65 de `simulateur.py`).

Dans le simulateur, un défaut est toujours le même objet : une copie de l'impulsion d'entrée, avec une amplitude tirée au hasard, placée au hasard entre les deux faces. Il y en a au plus un par pièce, et rien ne dit s'il s'agit d'une fissure, d'une bulle ou d'une inclusion, ni quelle est sa taille ou son orientation.

Dans une vraie pièce, ce sont justement ces propriétés qui déterminent l'écho. Une fissure orientée dans l'axe du faisceau renvoie peu de son, alors qu'une fissure perpendiculaire renvoie un écho fort. Plusieurs défauts peuvent coexister, et l'écho d'un défaut n'a pas forcément la même forme que l'impulsion de départ.

Un détail du code le montre bien : l'amplitude de l'écho du défaut et l'ombre qu'il fait sur l'écho de fond sont tirées séparément (lignes 62 et 64). Un petit défaut peut donc faire une grosse ombre, et inversement. Physiquement, un défaut plus grand devrait à la fois renvoyer plus de son et en bloquer davantage.

Conséquence : un réseau entraîné sur ce simulateur apprend à reconnaître un seul type d'écho, toujours de la même forme. Rien ne garantit qu'il reconnaîtrait les défauts variés d'une vraie pièce. C'est le point « un seul type de défaut » de la section 10 du guide.

D'autres simplifications comptent aussi, comme les classes équilibrées à 50/50 alors que les défauts sont rares en réalité, ou les étiquettes parfaites alors qu'en vrai il faut un expert pour les obtenir. Mais celle du défaut touche directement ce que le réseau apprend à détecter.

### Ma machine suffit-elle pour entraîner ?

J'ai un MacBook Pro 14 pouces avec une puce M5 et 24 Go de mémoire. Pour ce projet, c'est largement suffisant. Le CNN de `modele.py` a environ 15 000 paramètres et s'entraîne sur 6 000 signaux de 512 points : le guide annonce 1 à 2 minutes sur un ordinateur ordinaire. Les scripts tournent sur le processeur (CPU), car ils ne précisent aucun `device`.

PyTorch peut aussi utiliser la puce graphique des Mac Apple Silicon avec `device="mps"`. Pour un modèle aussi petit, ça n'accélère probablement pas grand-chose : copier les données vers la puce graphique coûte plus cher que le calcul lui-même. Une carte graphique devient indispensable pour des modèles de millions de paramètres ou des images en grand nombre, pas ici.

### Premier entraînement du CNN (`03_train_cnn.py`)

Le modèle a 15 169 paramètres. L'entraînement de 60 epochs a duré 240 secondes sur mon Mac, deux fois plus que les 1 à 2 minutes annoncées par le guide. Je ne sais pas encore pourquoi.

![Courbes d'entraînement](sorties/courbes_entrainement.png)

À l'epoch 1, la perte vaut 0,6934, presque exactement ln 2 = 0,6931 : c'est la perte d'un modèle qui répond 50/50 au hasard, et l'exactitude est de 0,501. Le réseau n'a encore rien appris. Entre les epochs 3 et 8, la perte stagne vers 0,64 et l'exactitude vers 0,65, puis l'apprentissage démarre vraiment vers l'epoch 10. L'exactitude de validation finit autour de 0,85 à 0,86, comme le guide l'annonce.

La perte de validation passe souvent sous la perte d'entraînement. Ce n'est pas une erreur : la perte d'entraînement est une moyenne calculée pendant que les poids changent au fil de l'epoch, alors que la validation est mesurée après, avec les poids déjà améliorés.

La courbe d'exactitude a des creux brusques, par exemple 0,785 à l'epoch 30 entre deux epochs à 0,83 et 0,84. Les mises à jour se font par mini-lots de 64 signaux, et chaque mini-lot donne une estimation un peu bruitée du gradient.

La mention « sauvegardé » apparaît quand la perte de validation bat son record. Le fichier `sorties/cnn_meilleur.pt` contient donc le modèle de l'epoch 55 (perte de validation 0,3380, exactitude 0,855), pas celui de l'epoch 60. On garde le meilleur modèle selon la validation plutôt que le dernier, parce que la validation peut se dégrader en fin d'entraînement.

### Entraîner sur la puce graphique du Mac (MPS)

Par curiosité, j'ai fait une variante du script d'entraînement, `03_train_cnn_mps.py`, qui utilise la puce graphique du Mac à travers le backend MPS de PyTorch. Le modèle, les données et la boucle sont identiques : seul le « device » change. Il faut y placer le modèle et les données avec `.to(device)`, copier les données une seule fois avant la boucle, ramener les poids sur le processeur avant de les sauvegarder, et attendre la fin des calculs (`torch.mps.synchronize()`) avant d'arrêter le chronomètre.

| Device | Durée (60 epochs) | Meilleure epoch | Perte val | Exactitude val |
|---|---|---|---|---|
| CPU | 240 s | 55 | 0,3380 | 0,855 |
| MPS | 16 s | 60 | 0,3322 | 0,864 |

La puce graphique est 15 fois plus rapide. Je m'attendais à moins, parce que le modèle est petit et que les mini-lots de 64 signaux laissent peu de travail au GPU à chaque étape. Ce résultat suggère surtout que le processeur était mal exploité : 240 s pour 15 000 paramètres, c'était lent.

Les résultats ne sont pas identiques aux décimales près, alors que la graine est la même. Le mélange des données est tiré par un générateur différent sur la puce graphique, et les calculs en virgule flottante n'y sont pas faits exactement dans le même ordre. Les deux entraînements arrivent au même niveau, autour de 85 à 86 % d'exactitude, ce qui est ce qui compte.

Plus tard, en relançant exactement le même entraînement sur MPS (modèle retenu, augmentation, sans entraînement adverse), j'ai obtenu un fichier `cnn_meilleur.pt` identique octet pour octet au précédent. Avec la même graine et la même machine, l'entraînement est donc reproductible ; c'est en changeant de matériel (CPU contre puce graphique) que les chiffres bougent.

Ensuite, j'ai gardé un seul script. Deux fichiers presque identiques auraient fini par diverger pendant les exercices. `03_train_cnn.py` choisit maintenant le device tout seul : la puce graphique si elle existe, sinon le processeur, donc il marche aussi sur une machine sans puce Apple. Il sauvegarde toujours dans `sorties/cnn_meilleur.pt`, avec les poids ramenés sur CPU, pour que les scripts 04 à 08 le retrouvent sans changement.

Au passage, j'ai découvert un piège : ajouter seulement la ligne `device = torch.device(...)` ne change rien. Elle choisit le device mais n'y déplace rien. Tant que le modèle et les données ne sont pas envoyés avec `.to(device)`, tout tourne sur le processeur, sans erreur ni avertissement. Et si on ne déplace que l'un des deux, PyTorch s'arrête avec une erreur, parce que le modèle et les données doivent être sur le même device.

### Les méthodes de référence et la limite d'une porte fixe

Le guide insiste sur une règle : toujours commencer par une méthode simple, pour savoir ce que le réseau doit battre. Sans point de comparaison, les 85 % du CNN ne voulaient rien dire. Avec la porte fixe à 65 % et la régression logistique à 50 %, je sais maintenant que le CNN apporte vraiment quelque chose.

La porte, c'est le réglage d'un technicien : une fenêtre de temps et un seuil, avec une alarme si un écho dépasse le seuil dans la fenêtre. Je pensais qu'on ne pouvait pas l'étendre à tout le signal à cause du nombre d'entrées. En fait, c'est l'écho de fond qui bloque : présent partout et plus grand qu'un défaut, il dominerait le maximum de n'importe quelle fenêtre qui l'inclut. Comme il se déplace d'une pièce à l'autre, une fenêtre fixe ne peut pas l'éviter tout en couvrant les défauts profonds. Le CNN n'a pas ce problème : il apprend à reconnaître la forme d'un défaut et à la distinguer de l'écho de fond, où qu'ils soient.

La régression logistique échoue pour une raison voisine. Elle donne un poids fixe à chaque position du signal, donc elle ne sait pas exprimer « il y a un petit écho quelque part ». C'est exactement ce que les convolutions savent faire, avec un même filtre qui glisse sur tout le signal.

En remplaçant le maximum par la somme dans la porte, l'exactitude tombe de 65 % à 56 %. J'en retiens qu'un score doit ressembler à ce qu'on cherche. Un défaut est un petit événement local, donc un score qui additionne toute la fenêtre noie ce petit événement dans le bruit accumulé sur les 130 échantillons. Le maximum isole l'événement le plus fort, ce qui est déjà plus proche de la question posée.

### Pourquoi on ne règle jamais rien sur le test

Je ne savais pas répondre à cette question. Le test sert à prédire comment le modèle se comportera sur des pièces nouvelles. Dès qu'on choisit quelque chose en regardant le test, même un simple seuil, le test n'est plus « nouveau » pour ce choix, et le score devient trop optimiste. C'est comme réviser avec les questions exactes de l'examen : la note ne dit plus si on maîtrise la matière.

D'où les trois jeux : l'entraînement pour apprendre les poids, la validation pour choisir les réglages, et le test pour mesurer une seule fois à la fin. Je retrouve la même logique dans `03_train_cnn.py`, qui garde le meilleur modèle selon la validation et ne touche jamais au test.

### Comment `modele.py` construit ses blocs de convolution

Le nombre de blocs n'est pas écrit en dur : la boucle `for c_out in canaux` crée un bloc [convolution, ReLU, max-pooling] par valeur du tuple `canaux`. Avec `(8, 16, 32, 32)`, il y a 4 blocs. Pour en ajouter un cinquième, il suffit d'ajouter une valeur, par exemple `(8, 16, 32, 32, 32)`. La variable `c_in` relie chaque bloc au suivant (les canaux de sortie de l'un deviennent les canaux d'entrée de l'autre), et la couche finale `nn.Linear(c_in, 1)` s'adapte toute seule au dernier nombre de canaux.

Chaque `MaxPool1d(2)` divise la longueur du signal par deux : 512, puis 256, 128, 64, 32 après 4 blocs, et 16 après un cinquième. Chaque bloc voit donc une portion du signal d'origine deux fois plus large que le précédent : c'est ce qui permet de passer d'une petite oscillation à un écho entier, puis à un écho avec son contexte.

Point d'attention : les scripts 04 à 08 créent le modèle avec `CNN1D()`, sans argument, puis chargent `sorties/cnn_meilleur.pt`. Si le fichier vient d'un modèle à 5 blocs, il faut que la valeur par défaut de `canaux` dans `modele.py` corresponde aussi à 5 blocs, sinon le chargement échoue.

### Ce que m'ont appris les exercices de l'étape 4

Changer un seul réglage à la fois et noter les chiffres dans un tableau m'a permis de voir l'effet de chacun.

La taille du réseau compte moins que je ne pensais. Avec 4 fois plus de paramètres, le modèle apprend plus vite mais plafonne au même niveau, vers 86 %. Ce plafond vient des données : certains défauts sont plus faibles que le bruit. Ajouter de la capacité au-delà de ce que les données permettent sert surtout à mieux coller à l'entraînement.

Le noyau, lui, a fait la différence : 15 au lieu de 9 donne le meilleur résultat (87,7 %). J'ai découvert le champ réceptif, la portion du signal qu'un neurone de la dernière couche voit. Il grandit avec le noyau et double à chaque max-pooling. Un filtre trop court (5 échantillons) ne voit même pas une oscillation complète de l'impulsion, qui dure environ 8 échantillons.

Le taux d'apprentissage peut tout casser. À 1e-2, le réseau reste à 50 % pendant 60 epochs ; à 1e-4, il avance si lentement qu'il n'atteint que 70 %. C'est exactement l'image de la montagne : des pas trop longs ou trop courts.

Le maximum global m'a piégé. J'ai prédit qu'il prendrait l'écho d'entrée, en oubliant que la normalisation l'efface. Le plus grand écho restant est l'écho de fond, présent partout, et le maximum le retient au détriment du petit défaut. Je dois relire ce que fait le prétraitement avant de prédire le comportement d'un modèle.

### Décision : je garde un noyau de 15

Après les exercices de l'étape 4, j'ai passé la valeur par défaut du noyau de 9 à 15 dans `modele.py` (`noyau=15`). C'est le réglage qui a donné le meilleur résultat sur la validation : perte 0,2941 et exactitude 0,877, contre 0,3322 et 0,864 avec 9. Le modèle passe de 15 169 à 25 201 paramètres, et l'entraînement de 16 à 20 secondes sur MPS. Tous les résultats à partir de l'étape 5 utilisent ce modèle. Comme ce choix a été fait sur la validation, le test reste une mesure honnête.

J'ai ensuite gardé aussi le cinquième bloc : `canaux=(8, 16, 32, 32, 32)`. Avec le noyau 15, il fait passer la meilleure perte de validation de 0,2941 à 0,2728, et l'exactitude de 0,877 à 0,883. Le modèle final a 40 593 paramètres et s'entraîne en 23 secondes sur MPS. Une réserve : chaque configuration n'a été entraînée qu'une fois, et un entraînement relancé donne des chiffres un peu différents (je l'ai vu entre CPU et MPS). L'écart sur l'exactitude (0,6 point) est du même ordre que cette variation ; celui sur la perte est plus net. Le modèle de référence pour la suite du projet est donc : 5 blocs, noyau 15, moyenne globale, LR 1e-3.

### À quoi sert le noyau d'une convolution

Le noyau est la fenêtre de poids d'un filtre : une petite suite de nombres (15 ici) que le réseau apprend. Le filtre fait glisser cette fenêtre le long du signal. À chaque position, il multiplie les 15 échantillons sous la fenêtre par ses 15 poids et additionne : c'est la même somme pondérée qu'un neurone, appliquée à toutes les positions avec les mêmes poids. Le résultat est grand là où le morceau de signal ressemble à la forme des poids, et petit ailleurs. Un filtre est donc un détecteur de forme, et le noyau fixe la longueur de la forme qu'il peut reconnaître.

Comme les mêmes poids servent à toutes les positions, le filtre reconnaît un écho où qu'il soit, ce que ne savaient faire ni la porte fixe ni la régression logistique. Le nombre de poids d'une convolution vaut canaux d'entrée × canaux de sortie × noyau : un noyau plus grand donne plus de paramètres, ce qui explique le passage à 25 201. Le `padding=noyau // 2` ajoute des zéros aux deux bouts pour que le signal garde sa longueur après la convolution.

### Reconnaître le surapprentissage sur des courbes

J'ai d'abord conclu au surapprentissage parce que la courbe de validation passait au-dessus de celle d'entraînement. Ce n'est pas le bon signe : la perte de validation est souvent un peu au-dessus, et c'est normal, puisque le modèle n'a jamais vu ces signaux. Le vrai signe est une tendance : la perte d'entraînement continue de baisser pendant que celle de validation stagne ou remonte, et l'écart entre les deux se creuse. Sur le modèle à 5 blocs, ça commence vers l'epoch 30. C'est aussi pour ça que le script garde le modèle de la meilleure epoch selon la validation, et pas celui de la dernière.

### Lire une matrice de confusion

La matrice de confusion range les 2 000 signaux du test selon deux questions : ce qu'ils sont vraiment (les lignes, « réel ») et ce que le modèle a répondu (les colonnes, « prédit »). Avec mon modèle (5 blocs, noyau 15) et un seuil de 0,5 :

| | prédit sain | prédit défaut |
|---|---|---|
| réellement sain (992) | 924 vrais négatifs | 68 faux positifs (fausses alarmes) |
| réellement défectueux (1 008) | 147 faux négatifs (défauts ratés) | 861 vrais positifs |

La diagonale contient les bonnes réponses, le reste les erreurs. Toutes les métriques se calculent à partir de ces quatre cases :

- exactitude = (924 + 861) / 2 000 = 0,893, la part de bonnes réponses ;
- rappel = 861 / 1 008 = 0,854, la part des défauts que le modèle trouve ;
- précision = 861 / (861 + 68) = 0,927, la part des alarmes qui sont de vrais défauts.

Les deux erreurs n'ont pas le même coût. Une fausse alarme fait recontrôler une pièce saine, ce qui coûte du temps. Un défaut raté laisse partir une pièce défectueuse, ce qui peut être grave. Ici, le modèle rate 147 défauts sur 1 008, soit environ un sur sept, alors qu'il ne fait que 68 fausses alarmes. Pour de l'inspection, on préférerait sans doute l'inverse : baisser le seuil pour rater moins de défauts, quitte à avoir plus de fausses alarmes.

### Un bon modèle peut être inutilisable quand les défauts sont rares

Sur le test, mon modèle a une AUC de 0,956. Au seuil choisi pour trouver au moins 95 % des défauts (0,14), il fait une précision de 0,713 : sept alarmes sur dix sont justifiées. Ça paraît correct.

Mais le test contient une pièce défectueuse sur deux, ce qui n'arrive jamais en production. Avec 1 % de pièces défectueuses, sur 10 000 pièces, le modèle trouverait 96 des 100 défauts, mais déclencherait aussi 3 901 fausses alarmes sur les 9 900 pièces saines. Seule une alarme sur 40 environ serait justifiée. Le taux de fausses alarmes (39 %) ne change pas, mais il s'applique maintenant à une immense majorité de pièces saines.

Ce que je retiens : le rappel et le taux de fausses alarmes ne dépendent pas de la proportion de défauts, la précision si. Avant de dire qu'un modèle est bon, il faut demander dans quelle proportion l'événement recherché se produit en vrai. Et le choix du seuil n'est pas une question technique : c'est une décision métier, qui pèse le coût d'un défaut raté contre celui d'une fausse alarme.

Je vois le seuil comme le réglage de sensibilité d'un détecteur de fumée, mais dans le sens inverse : plus le seuil est bas, plus le détecteur est sensible. À 0,14, une probabilité de défaut de 15 % suffit à déclencher l'alarme, donc il trouve presque tous les défauts mais sonne aussi pour beaucoup de pièces saines. Le seuil ne change rien au modèle : les probabilités qu'il calcule restent les mêmes, seule la ligne de décision se déplace. D'ailleurs, « sensibilité » est le nom que la médecine donne au rappel.

### Un résultat n'est pas un nombre exact : l'intervalle de confiance

Mon exactitude sur le test vaut 0,892, mais elle est mesurée sur un seul échantillon de 2 000 pièces. Avec 2 000 autres, elle aurait été un peu différente. Le bootstrap estime cette variation sans avoir besoin de nouvelles données : on tire 1 000 fois 2 000 signaux avec remise parmi les prédictions, on recalcule l'exactitude à chaque fois, et on garde les 95 % de valeurs centrales. J'obtiens [0,878 ; 0,905], soit environ ±1,4 point.

Ça change la façon de lire mes comparaisons de l'étape 4. Un écart de 0,6 point entre deux modèles est plus petit que l'incertitude : il pourrait venir du hasard. Pour comparer deux modèles sérieusement, il faut que l'écart dépasse nettement cette marge, ou répéter les entraînements avec plusieurs graines. Je présenterai désormais mes résultats sous la forme « 89,2 % ± 1,4 point » plutôt qu'avec un seul chiffre.

### L'augmentation de données, et une erreur d'étiquetage

L'augmentation de données consiste à perturber les signaux d'entraînement au hasard (ici un gain entre 0,7 et 1,3 et un peu de bruit) pour que le modèle apprenne à ne pas dépendre de ces variations. Résultat : aucune perte sur les signaux normaux, une forte amélioration face à une erreur d'étalonnage de 30 % (de 84 % à 89 %), un gain modeste face au bruit, et, sans l'avoir cherché, un peu plus de résistance à l'attaque adverse. Le modèle augmenté devient le modèle de référence.

J'ai d'abord envoyé mes deux sorties avec les étiquettes « avant » et « après » inversées. L'erreur a été repérée parce que le modèle sans augmentation devait redonner exactement les chiffres de l'étape 5 (0,892 et 0,854), et c'était la sortie « après » qui les affichait. Depuis, `05_robustesse.py` affiche en première ligne le fichier du modèle testé. Une leçon pratique : chaque résultat doit dire de quoi il est le résultat, et un chiffre déjà connu sert de point de contrôle.

L'augmentation a été réglée d'après les perturbations que le script teste ensuite (gain ±30 %, bruit jusqu'à 0,1). C'est un peu préparer le modèle à l'examen. Dans un vrai projet, on choisirait ces plages d'après ce qui arrive réellement sur le terrain.

### Ce projet relève de quel métier ?

Surtout du ML engineering : je construis un modèle de A à Z sur des données de capteur (signaux), je le compare à des méthodes de référence, je l'évalue honnêtement, je teste sa robustesse, puis je l'optimise et je le déploie (ONNX, quantification, C++, API). L'AI engineering, au sens où on l'emploie aujourd'hui, consiste plutôt à construire des applications au-dessus de grands modèles existants (LLM, RAG, agents), sans entraîner de réseau soi-même. Ce projet n'en fait pas.

Il touche aussi deux spécialités voisines. L'étape 6 relève de la sécurité des modèles (adversarial ML), qui rejoint ma formation en cybersécurité. Les étapes 8 et 9 (quantification, inférence en C++) relèvent de l'IA embarquée, qui compte beaucoup dans l'industrie, où les modèles tournent souvent sur des appareils plutôt que dans le cloud. Les bases que je travaille en parallèle (gradient, règle de la chaîne, rétropropagation) servent à tous ces métiers.

### Une attaque doit survivre à toute la chaîne de traitement

L'attaque FGSM d'origine modifiait tout le signal, y compris la zone morte que la normalisation remet à zéro. Dans un vrai système, cette partie de la perturbation serait effacée avant d'arriver au modèle. En la retirant, l'attaque reste aussi efficace (écarts de 0,2 à 1,5 point jusqu'à ε = 0,05). Elle passe donc entièrement par la partie utile du signal : la menace n'était pas un artefact du test. En sécurité des modèles, la bonne question n'est pas seulement « l'attaque marche-t-elle ? » mais « marche-t-elle encore après tous les traitements qu'un vrai attaquant devrait traverser ? ».

### L'entraînement adverse : une défense qui se paie

L'entraînement adverse consiste à entraîner le modèle sur des versions attaquées de ses propres signaux, recalculées à chaque étape. Ça marche contre l'attaque apprise : à ε = 0,02, mon modèle passe de 43,5 % à 70,4 %. Mais il perd 9 points de rappel sur les signaux normaux et devient moins robuste au bruit et à l'étalonnage. Il n'existe pas de robustesse gratuite : se protéger contre une menace peut affaiblir le modèle face aux autres, et surtout sur sa tâche principale.

Le choix dépend donc de la menace réelle. Pour une inspection, un défaut raté est le risque principal ; une attaque adverse suppose un attaquant capable de modifier le signal, ce qu'il faut d'abord démontrer. J'ai gardé le modèle sans entraînement adverse, en documentant l'essai.

J'ai aussi retenu une réserve de méthode : tester un modèle avec la même attaque que celle qui a servi à l'entraîner surestime la protection. Il faudrait une attaque différente et plus forte (PGD, plusieurs petits pas) pour conclure.

### Raisonnement de menace : qui pourrait attaquer un signal d'inspection ?

Ma première idée était un concurrent. C'est possible, mais peu probable : un concurrent a rarement accès à la chaîne d'inspection d'une autre entreprise, et il aurait plus simple à faire pour lui nuire. En reprenant la question par « qui a intérêt à ce qu'une pièce défectueuse passe ? », je vois d'autres profils, plus crédibles.

Qui :

- Un fournisseur qui produit les pièces, ou un sous-traitant chargé de les inspecter. S'il doit livrer à temps ou à un certain taux de conformité, faire passer des pièces défectueuses lui fait gagner de l'argent. La fraude à la qualité et les pièces non conformes sont des problèmes réels dans l'aéronautique.
- Un employé malveillant ou mécontent, qui a déjà un accès légitime aux appareils ou aux serveurs.
- Un acteur étatique ou un groupe organisé qui veut saboter une infrastructure critique (aviation, nucléaire, ponts, pipelines) en laissant passer des défauts qui céderont plus tard.
- À l'inverse, quelqu'un qui veut faire rejeter des pièces saines (fausses alarmes) pour bloquer une production : là, un concurrent devient plus plausible.

Où l'attaque peut se placer, de la sonde au modèle :

1. Sur le signal physique, à la sonde : très difficile, il faudrait produire une perturbation acoustique précise dans la pièce.
2. Dans l'appareil d'acquisition, par son logiciel ou son firmware : faisable pour quelqu'un qui accède à l'appareil ou à sa chaîne de mise à jour (attaque de la chaîne d'approvisionnement).
3. Pendant la transmission, entre l'appareil et le serveur qui fait tourner le modèle, si les données ne sont ni chiffrées ni signées.
4. À l'entrée du modèle, par exemple l'API Flask de l'étape 10, qui reçoit le signal en JSON : quiconque peut lui envoyer des requêtes peut lui soumettre un signal modifié.

Ce qu'il faut à l'attaquant :

- Pouvoir modifier le signal numérique après l'acquisition et avant le modèle (points 2 à 4). C'est la condition principale.
- Connaître le modèle, au moins en partie. Mon attaque FGSM est en boîte blanche : elle utilise les poids et le gradient du modèle. Sans eux, un attaquant peut entraîner un modèle de substitution sur des données semblables et espérer que l'attaque se transfère, ou interroger l'API de nombreuses fois pour deviner la direction à suivre.
- Savoir que la perturbation survit au prétraitement. L'exercice 2 montre que c'est le cas ici : l'attaque reste efficace même en laissant la zone morte à zéro.

Ce qui rend l'attaque adverse intéressante pour un attaquant, c'est qu'elle est discrète. Quelqu'un qui peut déjà modifier le signal pourrait simplement le remplacer par celui d'une pièce saine, mais un inspecteur pourrait remarquer l'anomalie. Une perturbation de 2 % est invisible à l'œil : le signal a l'air normal et le modèle se trompe.

Les défenses les plus utiles ne sont pas forcément dans le modèle : signer les signaux dès l'appareil pour détecter toute modification, chiffrer les transmissions, protéger l'accès à l'API et aux poids du modèle, journaliser chaque décision (comme le prévoit l'étape 10), et garder une revue humaine sur un échantillon de pièces. L'entraînement adverse ne vient qu'en complément, et seulement si la menace le justifie, vu son coût sur le rappel.

### ONNX : sortir le modèle de PyTorch, et vérifier après chaque conversion

ONNX est un format d'échange : un fichier qui décrit les couches et les poids du réseau indépendamment de PyTorch, que d'autres programmes et d'autres langages savent lire. ONNX Runtime est un moteur rapide qui exécute ce fichier. Le script vérifie que PyTorch et ONNX Runtime donnent les mêmes sorties : un export peut réussir sans erreur et quand même produire des résultats différents.

Le fichier ONNX fige aussi la forme des entrées. Seul le nombre de signaux par lot est variable ; un signal de 500 points est refusé, alors que le réseau saurait techniquement le traiter. Je trouve ce refus utile : une entrée qui ne ressemble pas aux données d'entraînement doit déclencher une erreur, pas une prédiction silencieusement fausse.

### Élagage et quantification : précision, vitesse, taille

L'élagage met à zéro les poids les plus petits. La quantification stocke chaque poids sur 8 bits (un entier avec un facteur d'échelle) au lieu de 32 bits à virgule. Le guide insiste sur une règle : toujours mesurer le triplet précision, vitesse, taille, jamais un seul des trois.

Avec mon modèle, la quantification est presque gratuite : 0,1 point d'exactitude en moins, deux fois plus rapide, trois fois plus petit (56 Ko). C'est ce qui permettrait de le faire tourner sur un petit processeur dans un appareil d'inspection.

L'élagage m'a surpris : mon modèle garde 88,5 % avec 30 % des poids supprimés, là où celui du guide s'effondre à 60 %. Un réseau plus gros a plus de redondance, donc il supporte mieux qu'on lui retire des poids. Ça donne un autre éclairage à mon choix de 5 blocs : les paramètres en plus n'apportaient presque rien en exactitude, mais ils rendent le modèle plus tolérant à la compression.

En réentraînant 3 epochs après l'élagage, le modèle récupère tout son niveau même avec 70 % des poids supprimés : les poids restants compensent ceux qui manquent. Et la quantification ne coûte rien en rappel au seuil d'inspection (27 défauts ratés en int8 contre 30 en float32). J'ai retenu qu'après toute compression, il faut vérifier la métrique qui compte pour le métier, ici le rappel au seuil choisi, et pas seulement l'exactitude, qui peut cacher un échange entre défauts ratés et fausses alarmes.

### Latence ou débit : traiter les signaux un par un ou par lots

En mesurant la latence par lots de 1, 16 et 128 signaux, j'ai vu que regrouper les signaux divise le temps par signal par trois. Chaque appel au modèle a un coût fixe, partagé entre tous les signaux du lot. Il faut donc distinguer deux besoins : la latence (le temps pour obtenir une réponse sur un signal précis, important si l'appareil doit réagir tout de suite) et le débit (le nombre de signaux traités par seconde, important pour analyser un balayage complet). On n'optimise pas de la même façon pour l'un et pour l'autre, et l'avantage de la quantification n'est pas le même : deux fois plus rapide pour un signal seul, seulement 20 % de mieux sur un gros lot.

### Questions encore ouvertes

- Installation réglée : toutes les dépendances du projet (torch 2.14.1, onnx, onnxruntime, flask, scikit-learn...) se sont installées sous Python 3.14, donc je garde cette version. La crainte de départ, que torch ou onnxruntime ne soient pas disponibles pour 3.14, ne s'est pas vérifiée.
- Comprendre pourquoi l'entraînement sur CPU prend 240 s au lieu des 1 à 2 minutes annoncées, alors que la puce graphique le fait en 16 s.
- Commencer la vidéo micrograd de Karpathy, maintenant que les vidéos 3Blue1Brown sont terminées.
