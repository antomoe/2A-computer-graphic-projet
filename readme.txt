# Projet 3D "Sea and underwater", Ensimag

## Propriété

Le sujet, les ressources de départ, le sujet du projet et les objets de obj/Fish
appartiennent à l'ENSIMAG - Grenoble INP.
Ce projet est un rendu étudiant réalisé à des fins d'étude par Alexandre Bouez,
An Xian & Yidi Zhu.

## Lancement du projet

Pour lancer le programme, il faut exécuter ‘viewer.py’.
Les prérequis sont Python3 avec les bibliothèques OpenGL, glfw, numpy, assimpcy, PIL.

## Contrôles en jeu

Les contrôles clavier sont illustrées dans ./images/controles.png.

### Tourner la caméra:

Avec la souris (Trackball) :
Tourner la caméra                     : clic gauche
Bouger la caméra                      : clic droit
(Dé)zoomer                            : molette

Avec le clavier :
Bouger dans une direction             : touches directionnelles
Zoomer                                : S
Dézoomer                              : F
Tourner la caméra vers le haut        : C
Tourner la caméra vers le bas         : D
Tourner la caméra vers la gauche      : X
Tourner la caméra vers la droite      : S

###  Contrôler le requin

Avec le clavier :
Bouger vers le haut                   : T
Bouger vers le bas                    : Y
Bouger vers la gauche                 : R
Bouger vers la droite                 : U
Avancer                               : I
Reculer                               : K
Tourner à gauche (PDV requin)         : J
Tourner à droite (PDV requin)         : L

La taille de l’écran est initialisée à 1280x960 mais peut être modifiée avec la souris.

## Rapport

### Modeling
Nous avons utilisé les modèles proposés ainsi que des modèles libres de droits trouvés en ligne. Nous avons ajouté plusieur méthodes de chargement des modèles animé avec leurs textures.
Pour créer des meshs plus complexe, nous avons ajouté des attributs nécessaires comme sa texture,
Nous avons ajouté un sol non plat, qui prend la forme d’un objet avec texture et sans animation que nous chargeons.
A cause de limites sur la vitesse de rendering, nous avons retirer certains des objets que nous avions ajouté afin que le rendu soit plus fluide.
Ainsi, nous avons du laisser de coté les herbes et algues que nous avions ajouté.

### Rendering
Nous avons utilisé un effet de lumière qui varie avec le temps.
Nous avons creer un skybox qui représente un environnement infini avec un horizon inatteignable.
Nous avons utilisé plusieurs textures différentes, notamment pour les objets animés ainsi que la skybox.
Nous avons ajouté un effet fog. Plus les objets sont éloignés, plus un effet de teinte bleu s’applique, masquant leur texture. Cet effet simule la vision limitée sous l’eau.
Nous avons ajouté un effet de transparence appliqués à des bulles.

### Animation
Nous choisissons pour chaque poisson au hasard des valeurs pour les positions et l’angle de tour afin de les faire se déplacer.
Les poissons sont également animés par des fichiers .fbx individuels.
Nous avons ajouté un requin avec animation squelette que nous pouvons contrôler par keyboard.
