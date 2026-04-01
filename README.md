# Projet Framework d'Entraînement PyTorch

Le projet est un framework minimal d'entraînement en PyTorch, conçu pour être hautement configurable et extensible afin de tester rapidement différentes configurations d'apprentissage profond. L'accent a été mis sur la qualité logicielle, la séparation des responsabilités et la reproductibilité des expériences plutôt que sur l'obtention de performances optimales.

La tâche de référence implémentée est la classification d'images sur le jeu de données CIFAR-10. Ce dataset a été choisi car il est équilibré (10 classes avec le même nombre d'exemples par classe).

Pour la classification d'image, on utilise les réseaux convolutifs (CNN).
Lorsque on utilise les CNN, on peut rencontrer un certain nombre de problème, notamment avec le input_size_linear. On a 2 solutions pour cela : faire passer une image avant l'entrainement ou modifier la classe avec LazyLinear.

## Fonctionnalités Implémentées

Conformément aux exigences du projet, ce framework permet dynamiquement de :
*  **Changer l'architecture** du réseau de neurones : On peut modifier le nombre de couches cachés, le nombre de input channels.
*  **Modifier les hyper-paramètres** d'entraînement : On peut modifier le nombre d'epochs, la patience de l'early stopping, la taille du batch, le "debug_mode" pour choisir si on entraîne avec tous le jeu de données de CIFAR-10;
*  **Choisir l'optimiseur** et le **scheduler de taux d'apprentissage**.
*  **Sélectionner la fonction de perte** : On peut choisir entre une fonction CrossEntropy ou MSE.
*  **Activer des augmentations de données** : Pour activer l'augmentation de données, compiler le script train.py en ajoutant "augmentation = basic"
*  **Faire de l'early stopping** pour prévenir le surapprentissage.

---

##  Architecture du Projet
Le code est structuré de manière claire et modulaire pour garantir une séparation stricte .

```text
├── configs/                 # Fichiers de configuration (Hydra).
│   ├── config.yaml          # Point d'entrée principal.
│   ├── model/               # Architectures réseaux (ex: resnet.yaml)
│   ├── optimizer/           # Optimiseurs (ex: adam.yaml, sgd.yaml)
│   ├── scheduler/           # Schedulers (ex: step_lr.yaml)
│   ├── loss/                # Fonctions de perte (ex: cross_entropy.yaml)
│   └── data/                # Paramètres CIFAR-10 et augmentations
├── src/                     # Code source du projet.
│   ├── models/              # Classes des réseaux de neurones
│   ├── data/                # Dataloaders et transformations
│   ├── training/            # Boucle d'entraînement, validation, early stopping
│   └── utils/               # Loggers (W&B) et métriques
├── train.py                 # Script principal d'entraînement
├── sweep.py                 # Script pour l'optimisation Optuna
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation utilisateur.
```

---
## Installation


## Commande bash pour lancer l'entrainement

---
ok le dossier augemntation dans config (yelman) correspond au parametrere qu'on utilise dans data/dataloader.py --> on parle notamment du batch_size et des parametres pour l'augmentation de donnée (notamment pour le transform mais pas forcement utile a faire pour l'instant ca sera que des trucs antoine deleforge & des modifs yelman a faire ensuite)
