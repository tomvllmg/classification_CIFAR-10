# Projet Framework d'Entraînement PyTorch

Ce projet a été réalisé dans le cadre du projet d'apprentissage profond évalué par Sylvain Faisan. 

Il propose un framework minimal d'entraînement en PyTorch, conçu pour être hautement configurable et extensible afin de tester rapidement différentes configurations d'apprentissage profond. L'accent a été mis sur la qualité logicielle, la séparation des responsabilités et la reproductibilité des expériences plutôt que sur l'obtention de performances optimales.

La tâche de référence implémentée est la classification d'images sur le jeu de données CIFAR-10. Ce dataset a été choisi car il est équilibré (10 classes avec le même nombre d'exemples par classe).

Pour la classification d'image, on utilise les réseaux convolutifs (CNN).
Lorsque on utilise les CNN, on peut rencontrer un certain nombre de problème, notamment avec le input_size_linear. On a 2 solutions pour cela : faire passer une image avant l'enrtrainement ou modifier la classe avec LazyLinear.

## Fonctionnalités Implémentées

Conformément aux exigences du projet, ce framework permet dynamiquement de :
* 🧠 **Changer l'architecture** du réseau de neurones.
* ⚙️ **Modifier les hyper-paramètres** d'entraînement.
* 📉 **Choisir l'optimiseur** et le **scheduler de taux d'apprentissage**.
* 🎯 **Sélectionner la fonction de perte** (loss).
* 🖼️ **Activer ou modifier des augmentations de données**.
* 🛑 **Faire de l'early stopping** pour prévenir le surapprentissage.

### Outils intégrés
* **PyTorch pur :** Le framework est codé sans surcouche de haut niveau (pas de pytorch-lightning, fastai, etc.).
* **Hydra :**  Gestion avancée et modulaire des fichiers de configuration.
* **Weights & Biases (W&B) :** Suivi des expériences et sauvegarde en temps réel des courbes d'apprentissage.
* **Optuna :** Optimisation automatique des hyper-paramètres.

---

## 📁 Architecture du Projet
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


ok le dossier augemntation dans config (yelman) correspond au parametrere qu'on utilise dans data/dataloader.py --> on parle notamment du batch_size et des parametres pour l'augmentation de donnée (notamment pour le transform mais pas forcement utile a faire pour l'instant ca sera que des trucs antoine deleforge & des modifs yelman a faire ensuite)
