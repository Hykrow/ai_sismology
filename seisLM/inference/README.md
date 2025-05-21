# Utilisation du dossier **inference/**

Ce guide s'applique lorsque vous utilisez des données simulées (HEMEW-3D). Pour ETHZ, remplacez simplement les noms de scripts sans le préfixe `simulated_`.

## Pré-requis

* Avoir généré les bases de données et tâches via `simulated_generate_tasks23.py`.
* Avoir placé les données dans le bon dossier (`simulated_data/velocities/`).

## Scripts disponibles

| Script                              | Description                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `simulated_generate_tasks23.py`     | Génère les tâches pour le fine-tuning ou l'inférence. |
| `simulated_generate_predictions.py` | Lance l'inférence sur le modèle seisLM.               |

> **Note** : Pour utiliser la base ETHZ, exécutez `generate_tasks23.py` puis `generate_predictions.py` sans le préfixe `simulated_`.

## Usage

1. **Génération des tâches**

   ```bash
   # Depuis la racine du projet seisLM
   python inference/simulated_generate_tasks23.py
   ```

2. **Inference du modèle**

   ```bash
   python inference/simulated_generate_predictions.py
   ```

Assurez-vous d'exécuter ces commandes depuis le dossier `seisLM`, conformément à l'architecture du projet.
