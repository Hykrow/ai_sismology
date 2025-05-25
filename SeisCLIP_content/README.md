# SeisCLIP : Démarche générale à suivre pour pouvoir tester SeisCLIP sur la base HEMEW-3D

Travail effectué par **Hamza** et **François**.
**Tous les chemins sont indiqués à partir du dossier `SeisCLIP`**

## Importer les données depuis HEMEW-3D
  -  **Télécharger les données sur HEMEW-3D**

Placer les traces dans `MIFNO/data/velocity/` et les métadonées dans `MIFNO/data/metadata/`

##  Préparation des images
-  **Extraire les fichiers .hdf5 au format STEAD**

Utiliser `MIFNO/data/conversion_format_stead.py` 
-  **Préparer les spectrogrammes**

Utiliser `MIFNO/data/conversion_spectogramme.py`
-  **Préparer les embedings**

Utiliser `spect_to_embed.py`
##  Préparation du texte

- **Extraire les métadonnées et les mettre au format STEAD**

Utiliser `metadata_format_stead.py`
- **Préparer les embedings**

Utiliser `embed_text.py`

## Tester le modèle sur les données préparées

- **Tester le modèle**

**Placer le répertoire courant du terminal à `SeisCLIP` pour ne pas avoir de problème avec les chemins relatifs indiqués dans le code.**
Appeler `efficacite_SeisCLIP.py`
  
