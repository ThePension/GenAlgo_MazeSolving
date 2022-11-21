# GenAlgo_MazeSolving

## Fonctionnement de l'algorithme génétique
### Définition d'un gène
Un gène peut prendre les 4 valeurs suivantes :
- 0 = Déplacement vers la gauche,
- 1 = Déplacement vers la droite,
- 2 = Déplacement vers le haut,
- 3 = Déplacement vers le bas

### Définition d'un chromosome (ADN)
Un chromosome compose un individu ; on peut faire une analogie avec l'ADN qui compose l'humain.
Un chromosome est une suite, d'une certaine longueur, de gènes. Un chromosome, à partir d'une position initiale, peut être appliqué afin d'obtenir une position finale.
Cela est fait dans la fonction de `compute_complete_valid_path` décrite ci-dessous.


### Récupération du chemin à partir de l'ADN (compute_complete_valid_path)
Afin de générer le chemin lié au chromosome, il suffit, pour chaque gène, d'appliquer le déplacement à la position actuelle, et d'ajouter celle-ci dans une liste pour garder une trace des positions empruntées.

Au départ, si appliquer la gène actuelle résidait en une position illégale (mur, en dehors du tableau), celle-ci n'était juste pas appliquée. Une autre approche a été mise en place afin de prévenir les cul-de-sac.


#### Prévenir les cul-de-sac
Si l'application d'une gène génère une position illégale, la gène actuelle est alors transformée en un autre déplacement, jusqu'à ce que celui-ci donne une position légale.

Si aucun des quatre types de gène ne permet d'obtenir une position légale, alors la position actuelle est un cul-de-sac, et sera considérée comme un mur jusqu'à la fin de la génération du chemin, pour ce chromosome uniquement.


### Récupération du chemin le plus court (compute_subpath)
La première étape consiste à retirer les positions dupliquées consécutives : par exemple, si une partie du chemin est composé de `[..., (1, 2), (1, 2), (1, 3), ...]`, il deviendra alors `[..., (1, 2), (1, 3), ...]`.

Par la suite, une fois la redondance éliminée, la fonction regarde si la position souhaitée (*target*) est contenue dans le chemin. Si c'est le cas, elle retourne le chemin à partir de la dernière position `(0, 0)` se trouvant avant la *target*, jusqu'à la première *target*.

Si la *target* ne se trouve dans le chemin, le même raisonnement est appliqué, mais en prenant comme *target* la position la plus proche de la *target* dans le chemin (distance euclidienne).


### Fonction de Sélection

### Fonction de Crossover

### Fonction de Mutation

### Fonction de Fitness

## Sources
- https://science.donntu.edu.ua/ipz/sobol/links/maze_solving.pdf
- https://www.researchgate.net/publication/233786685_A-Mazer_with_Genetic_Algorithm
- https://www.diva-portal.org/smash/get/diva2:927325/FULLTEXT01.pdf
