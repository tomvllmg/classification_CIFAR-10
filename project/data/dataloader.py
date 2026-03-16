import torch
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(cfg_data):
    data_dir = cfg_data.get("data_dir", "./data")
    batch_size = cfg_data.get("batch_size", 128)
    num_workers = cfg_data.get("num_workers", 2)
    use_aug = cfg_data.get("use_augmentation", True) # Récupération du choix YAML

    # --- 1. Définition des transformations ---
    
    # Transformation de base (toujours nécessaire)
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    # Ajout de l'augmentation si demandé
    aug_transform = []
    if use_aug:
        aug_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    
    # On compose les deux listes
    transform_train = transforms.Compose(aug_transform + base_transform)
    transform_test = transforms.Compose(base_transform)

    # --- 2. Chargement et Split ---

    # On charge le dataset complet d'entraînement
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    # Split 80/20
    n_train = int(len(full_train_dataset) * 0.8)
    n_val = len(full_train_dataset) - n_train
    
    # On utilise un générateur avec une seed fixe si on veut de la reproductibilité
    train_subset, val_subset = random_split(
        full_train_dataset, [n_train, n_val], 
        generator=torch.Generator().manual_seed(42)
    )

    # ASTUCE : Pour que la validation n'ait PAS d'augmentation de données
    # Le random_split garde le 'transform' du dataset parent. On force celui de test sur la validation.
    val_subset = copy.deepcopy(val_subset) # Optionnel mais propre
    val_subset.dataset.transform = transform_test 
    # Attention: dans certains cas complexes, changer val_subset.dataset.transform impacte train_subset.
    # Si c'est le cas, la méthode la plus robuste est de créer deux datasets identiques au départ 
    # et d'appliquer les indices du split.

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # --- 3. Création des DataLoaders ---

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
