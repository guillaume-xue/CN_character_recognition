# Reconnaissance de Caractères Chinois

Un projet de machine learning pour la reconnaissance automatique de caractères chinois manuscrits et imprimés en utilisant des modèles de classification.

## Description

Ce projet entraîne et évalue plusieurs modèles de classification (SVM, Régression Logistique, Random Forest) pour reconnaître les caractères chinois. Il utilise deux ensembles de données:
- **Chinese MNIST** : caractères imprimés
- **CASIA-HWDB** : caractères manuscrits

Les images sont prétraitées et les caractéristiques sont extraites usando la méthode HOG (Histogram of Oriented Gradients) avant l'entraînement des modèles.

## Structure du Projet

```
├── main.py                    # Point d'entrée principal
├── data/
│   ├── chinese_mnist.csv      # Métadonnées du dataset Chinese MNIST
│   ├── chinese_handwriting.csv # Métadonnées du dataset CASIA-HWDB
│   ├── train_model/           # Modèles sauvegardés
│   └── load_data/             # Données prétraitées en cache
├── src/
│   ├── data_loader.py         # Chargement et prétraitement des données
│   ├── features.py            # Extraction des features HOG
│   └── model.py               # Entraînement et évaluation des modèles
├── Makefile                   # Tâches d'automatisation
└── README.md                  # Ce fichier
```

## Installation

### Prérequis
- Python 3.7+
- pip ou conda

### Étapes d'installation

1. Cloner le repository:
```bash
git clone <repository-url>
cd CN_character_recognition
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Télécharger et extraire les données:
```bash
make build
```

## Utilisation

### Entraîner et évaluer les modèles

```bash
python main.py
```

Cela va:
1. Charger les ensembles de données
2. Entraîner les trois modèles (SVM, Régression Logistique, Random Forest)
3. Évaluer et afficher les performances

### Nettoyage

Pour nettoyer les fichiers générés :
```bash
make clean
```

## Modules

### `data_loader.py`
Charge les images depuis les deux datasets et les prétraite:
- Lecture des fichiers CSV de métadonnées
- Localisation et chargement des images
- Extraction des features HOG
- Sauvegarde en cache pour optimiser les performances

### `features.py`
Contient les fonctions de traitement d'images:
- **Binarization** : Otsu thresholding
- **Cropping** : Suppression des zones blanches
- **Resizing** : Redimensionnement à 128x128
- **HOG Features** : Extraction des descripteurs HOG

### `model.py`
Entraîne et évalue les modèles de classification:
- **SVM** : Support Vector Machine avec optimisation des hyperparamètres
- **LogisticRegression** : Régression logistique
- **RandomForest** : Forêt aléatoire

## Structure des Données

Deux datasets sont supportés:

### 1. Chinese MNIST Dataset
- Format: Images JPG
- Structure: `data/data/input_{index1}_{index2}_{index3}.jpg`
- Métadonnées: `data/chinese_mnist.csv`

### 2. CASIA-HWDB Dataset
- Format: Images PNG
- Structure: `data/chinese-handwriting/CASIA-HWDB_{split}/{split}/{character}/{index}.png`
- Métadonnées: `data/chinese_handwriting.csv`

## Résultats et Performances

Les modèles sont évalués sur l'ensemble de test avec les métriques standard de classification. Les features HOG sont efficaces pour capturer les motifs structurels des caractères.

## Technologies Utilisées

- **scikit-learn** : Training et évaluation des modèles
- **OpenCV** : Traitement d'images
- **scikit-image** : Extraction des features HOG
- **NumPy** : Manipulation des données
- **Matplotlib** : Visualisation

## License

Voir le fichier `LICENSE` pour les détails.

## Auteur

Guillaume

## Notes

- Les modèles entraînés sont sauvegardés dans `data/train_model/` pour réutilisation ultérieure
- Les données prétraitées sont mises en cache dans `data/load_data/` pour accélérer les itérations
- L'extraction des features HOG peut être longue sur de grands datasets - la mise en cache est utilisée automatiquement
