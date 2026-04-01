# Projet Framework d'Entraînement PyTorch
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
├── configs/                 # Fichiers de configuration.
│   ├── config.yaml          # Point d'entrée principal.
│   ├── model/               # Architectures réseaux.
│   ├── optimizer/           # Optimiseurs.
│   ├── scheduler/           # Schedulers.
│   ├── loss/                # Fonctions de perte.
│   └── data/                # Paramètres CIFAR-10 et augmentations.
├── src/                     # Code source du projet.
│   ├── models/              # Classes des réseaux de neurones.
│   ├── data/                # Dataloaders et transformations.
│   ├── training/            # Boucle d'entraînement, validation, early stopping.
│   └── utils/               # Loggers (W&B) et métriques.
├── train.py                 # Script principal d'entraînement.
├── sweep.py                 # Script pour l'optimisation Optuna.
├── requirements.txt         # Dépendances Python.
└── README.md                # Documentation utilisateur.
```
