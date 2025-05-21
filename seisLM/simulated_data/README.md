# Utilisation du dossier **simulated\_data/**

Ce dossier contient tout le nécessaire pour préparer, annoter et comparer les données simulées (HEMEW-3D).

## Structure

* **velocities/**

  * Placez-y les fichiers `.zip` de la base de données HEMEW (autant que nécessaire). Exemples :

    * `velocity100000-100099.zip`
    * `velocity100200-100299.zip`

* **Scripts**

  1. **zip\_data\_conversion.py**

     * Convertit les archives `.zip` de vitesses en un fichier HDF5 unifié.

     ```bash
     python simulated_data/zip_data_conversion.py
     ```

  2. **zip\_data\_annotate\_csv\_sta.py**

     * Calcule les indicateurs STA/LTA et génère un fichier CSV d’annotations.

     ```bash
     python simulated_data/zip_data_annotate_csv_sta.py
     ```

* **Comparaisons de résultats**

  * **comparison\_sta\_phasenet/**

    * Compare les résultats STA et PhaseNet.
    * Avant d’exécuter la comparaison, lancez la détection PhaseNet :

      ```bash
      python simulated_data/phasepick_simulated_phasenet.py
      ```

  * **comparison\_sta\_seislm/**

    * Compare les résultats STA et seisLM (ou tout autre modèle configuré dans `seisLM/configs`).
    * Assurez-vous d’avoir effectué l’inférence sixLM au préalable :

      ```bash
      python inference/simulated_generate_predictions.py
      ```
