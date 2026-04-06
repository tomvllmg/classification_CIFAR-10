FRAMEWORK MINIMAL D'ENTRAINEMENT PYTORCH POUR CIFAR-10

DESCRIPTION
-----------
L'objectif de ce projet est de concevoir et implémenter un framework minimal d'entraînement en PyTorch, configurable et extensible, permettant de tester rapidement différentes configurations d'apprentissage profond. L'accent est mis sur la qualité logicielle, la structuration du code, la séparation des responsabilités et la reproductibilité des expériences.

La tâche de référence est la classification d'images sur le jeu de données CIFAR-10.

Ce framework respecte la contrainte d'utiliser du PyTorch pur tout en intégrant nativement :
- Hydra pour la gestion dynamique des configurations YAML et l'instanciation.
- Weights & Biases (W&B) pour le suivi et la sauvegarde des courbes.
- Optuna pour une optimisation automatique des hyper-paramètres.


STRUCTURE DU PROJET
-------------------
L'architecture logicielle est modulaire et facilement extensible.

/config/                  - Configurations Hydra
  /augmentation/          - basic.yaml, none.yaml
  /loss/                  - cross_entropy.yaml, mse.yaml
  /model/                 - cnn.yaml
  /optimizer/             - adam.yaml, sgd.yaml
  /scheduler/             - cosine.yaml, step_lr.yaml
  config.yaml             - Configuration principale (Defaults)
/data/                    - dataloader.py
/losses/                  - build_loss.py
/model/                   - cnn.py, build_model.py
/optimizers/              - build_optimizer.py
/schedulers/              - build_schedulers.py
/utils/                   - early_stopping.py
train.py                  - Script d'entraînement principal
optuna_opti.py            - Script d'optimisation (Optuna)
test.py                   - Script d'évaluation sur le jeu de test

CONFIGURATION PAR DEFAUT
------------------------
La configuration par défaut (définie dans le fichier config/config.yaml) lance un entraînement en mode debug (debug_mode: true) sur un modèle CNN de base (2 couches cachées, 16 canaux initiaux). L'apprentissage s'effectue sur 15 époques avec des lots de 64 images, en utilisant l'optimiseur Adam (learning rate de 0.001), la fonction de perte Cross-Entropy et le scheduler StepLR (réduction du learning rate tous les 10 pas). Aucune augmentation de données n'est appliquée par défaut, et l'arrêt précoce (early stopping) interviendra si aucune amélioration de la précision n'est observée pendant 10 époques consécutives.


UTILISATION ET CONFIGURATION (Lignes de Commande Hydra)
-------------------------------------------------------
Grâce à Hydra, tous les paramètres du framework peuvent être modifiés directement depuis le terminal via la ligne de commande. La configuration par défaut est définie dans config/config.yaml.

ATTENTION : Par défaut, le paramètre data.debug_mode est réglé sur True pour permettre des tests rapides sur un sous-ensemble des données. Pour un entraînement réel et complet, on doit le passer à False.

1. Entraînement standard
Pour lancer un entraînement avec la configuration par défaut (CNN, Adam, CrossEntropy, StepLR, mode debug actif) :
> python train.py

Pour lancer l'entraînement complet sur tout le dataset :
> python train.py data.debug_mode=false

2. Changer un module complet (Hydra)
On peut remplacer une configuration yaml en appelant le nom du fichier YAML (sans le .yaml).

- Changer l'optimiseur (Adam -> SGD) :
> python train.py optimizer=sgd

- Changer le scheduler (StepLR -> Cosine) :
> python train.py scheduler=cosine

- Activer l'augmentation de données "basic" :
> python train.py augmentation=basic

- Combiner plusieurs changements :
> python train.py optimizer=sgd scheduler=cosine augmentation=basic

3. Modifier des hyper-paramètres spécifiques à la volée
On peut modifier une valeur précise contenue dans les fichiers YAML en utilisant la syntaxe groupe.parametre=valeur.

- Modifier les paramètres d'entraînement (Epochs, Patience, Batch_size) :
> python train.py training.epochs=50 training.patience=15 data.batch_size=128

- Modifier l'architecture du modèle (ex: 3 couches cachées, 32 canaux) :
> python train.py model.param.nb_hidden_layers=3 model.param.num_channels1=32

- Modifier les hyper-paramètres de l'optimiseur (ex: Learning rate pour Adam) :
> python train.py optimizer.param.lr=0.005 optimizer.param.weight_decay=0.001

- Changer de module ET modifier ses paramètres en même temps :
> python train.py optimizer=sgd optimizer.param.lr=0.05 optimizer.param.momentum=0.95

4. Weights & Biases
Pour bien identifier vos runs sur l'interface W&B, on peut renommer l'expérience à la volée :
> python train.py wandb.run_name="Essai_SGD_LR0.05_AugmentationBasic"


OPTIMISATION AUTOMATIQUE AVEC OPTUNA
------------------------------------
Pour rechercher automatiquement les meilleurs hyper-paramètres (le learning rate et le weight decay sont explorés par défaut), exécutez le script dédié :

> python optuna_opti.py data.debug_mode=false

Le meilleur modèle trouvé lors des essais sera automatiquement sauvegardé sous le nom best_optuna_model.pt.


EVALUATION DU MODELE (TEST)
---------------------------
Une fois un modèle entraîné (soit par train.py qui génère model_classif_val_train.pt, soit par optuna_opti.py qui génère best_optuna_model.pt), on peut évaluer ses performances sur le jeu de test final. 

Par défaut, test.py cherche le meilleur modèle issu d'Optuna.
> python test.py data.debug_mode=false
