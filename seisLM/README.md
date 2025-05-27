# Utilisation de seisLM (+ finetunage) et PhaseNet sur HEMEW-3D, base de données simulée

## Structure du projet

* **inference/** : dossier pour effectuer les prédictions avec seisLM.
* **pretrained/** : emplacement pour déposer le fichier `pretrained.ckpt`.

  * Téléchargez le pretrained ici : [https://github.com/liutianlin0121/seisLM](https://github.com/liutianlin0121/seisLM)
* **simulated\_data/** : génération des bases de données, prédictions PhaseNet, calcul STA/LTA.
* **src/phasepick/phasepick\_run.py** :

  * Modifié spécialement pour utiliser la base de données simulée (finetuning sur la BDD simulée HEMEW).
  * Compatible avec PhaseNet, EQTransformer, GDPick et seisLM.
  * Utilise les fichiers de configuration adaptés situés dans `seisLM/configs/phasepick` (la base de données est interchangeable, la simulée sera utilisée).

## Usage

1. **Préparez les données**

   * Placez les archives `.zip` contenant les données de vitesse dans `simulated_data/velocities/`.

2. **Générez la base de données**

   * Depuis la racine du projet seisLM, lancez les scripts de génération :

     ```bash
     # Génération des tâches de simulation
     python inference/simulated_generate_tasks23.py

     # Génération des prédictions du modèle simulé
     python inference/simulated_generate_predictions.py
     ```

   * **Attention** : ces commandes doivent être exécutées depuis le dossier `seisLM` en raison de la manière dont le projet est structuré.

3. **Comparez les résultats**

   * Utilisez le script de comparaison :

     ```bash
     python simulated_data/comparison_sta_seislm.py
     ```

> **Note** : L’exécution depuis `seisLM` est requise en raison de l’architecture interne du projet.

> **Note** : Il y a d'autres README dans les dossiers importants.

## Scripts de PhasePick

* **src/phasepick/phasepick\_run.py** :

  * Modifié spécialement pour utiliser la base de données simulée (finetuning sur la BDD simulée HEMEW).
  * Compatible avec PhaseNet, EQTransformer, GDPick et seisLM.
  * Utilise les fichiers de configuration adaptés situés dans `seisLM/configs/phasepick` (la base de données est interchangeable, la simulée sera utilisée).

* **src/phasepick/phasepick\_run\_only\_head\_trainable.py** :

  * Fontionne de la même manière que **phasepick_run** mais en posant uniquement les têtes des transformers comme paramètre à entrainer. (Comme suggéré par le client)

* **src/phasepick/phasepick\_run\_classic.py** :

  * Utilisé pour le fine-tuning avec les bases de données associées aux configs (non simulées).

### Exemple d'utilisation

Pour lancer un fine-tuning ou une inférence avec l'un de ces scripts :

```bash
python src/phasepick/phasepick_run.py --config /home/noam/ai_project/seisLM/seisLM/configs/phasepick/ethz_phasenet.json --save_checkpoints
```

## Évaluation

Dans `seisLM/evaluation`, vous trouverez `pick_eval_debug.py`, équivalent de `pick_eval.py` mais adapté aux données simulées. Il gère le dataset spécifique et offre davantage de sorties pour le troubleshooting si nécessaire.
