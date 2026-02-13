# Détection de Glaucome par Classification d'Images Fundus

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Fusion](https://img.shields.io/badge/Dataset-RIM--ONE%20+%20EyePACS-blue)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  # Adapter avec lien réel

## Description

Projet de classification binaire (glaucome vs sain) sur images fundus oculaires. Utilisation d'un dataset fusionné (RIM-ONE + EyePACS) avec augmentations Albumentations pour lutter contre l'overfit. Modèle : MobileNetV3-Small pré-entraîné, entraîné sur CPU pour accessibilité.

Ce projet montre des techniques de vision par ordinateur en santé, avec focus sur l'efficacité sur datasets limités.

## Résultats clés

- Meilleure Val Acc : 90.78% (après 8 epochs)
- Train Acc finale : 87.12%
- Modèle léger : 1.5M paramètres, adapté pour déploiement mobile.

![Exemple Fundus](Glaukompapille2.jpg)  
*Exemple d'image fundus avec glaucome (source : Wikimedia Commons)*

## Technologies

- **Langage** : Python 3.10+
- **Librairies** : PyTorch, Torchvision, Albumentations, Tqdm, NumPy
- **Dataset** : Fusion RIM-ONE + EyePACS (~9k images total, équilibré train/val/test)

## Installation

1. Clone le repo :
   ```bash
   git clone https://github.com/SiremKaci/glaucoma-detection.git
   cd glaucoma-detection
Installe les dépendances :
pip install -r requirements.txt
Prépare le dataset fusionné (voir Fuuusion.ipynb pour fusion).
Lance l'entraînement :
jupyter notebook Taches.ipynb
