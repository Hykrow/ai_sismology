import pandas as pd
mat = pd.read_csv("data/metadata/metadata_materials.csv")          # délimiteur par défaut ','
src = pd.read_csv("data/metadata/source_properties.csv")


# Exemple si l’identifiant commun est 'index'
df = mat.merge(src, on="index", how="inner")
df.to_csv("data/metadata/metadata_complete.csv", index=False)
print("Fichier fusionné :", df.shape, "lignes")

