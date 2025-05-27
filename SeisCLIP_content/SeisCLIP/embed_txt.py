import os
import numpy as np
import math

# Forcer torch à utiliser ton répertoire perso pour les checkpoints
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')


from Pretrain.model_seismic_clip_two_branch import AUDIO_CLIP
import torch, h5py, pandas as pd, torch.nn.functional as F
import torchaudio
from pathlib import Path

# chemins ---------------------------------------------------------------------
CSV_META = Path("MIFNO/data/my_velocity_metadata.csv")
OUT_NPY  = Path("text_emb_384.npy")

#CSV_META = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data/my_velocity_metadata.csv")
#OUT_NPY  = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/text_emb_384.npy")


#test sur les 100 premières lignes avec les ondes p et s 
#CSV_META = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data/metadata_with_SandP_arrivals.csv")
#OUT_NPY  = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/text_emb_384.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) ── charger SeisCLIP (branche texte uniquement) ---------------------------
model = AUDIO_CLIP(device_name=device,
                   embed_dim=384,  text_input=8, text_width=512, text_layers=2,
                   spec_fdim=50,   spec_tdim=600,
                   spec_fstr=10,   spec_tstr=10,
                   spec_model_size="small224",
                   imagenet_pretrain=True).to(device)

state_dict = torch.load('checkpoints/SeisCLIP_STEAD_50x600.ckpt', map_location=device)
model.load_state_dict(state_dict, strict=False)   # on ignore la partie audio
model.eval()

# ---------------------------------------------------------------------------
# 2) CSV → tensor (N, 8)   • nouvelle mise à l’échelle 0-1
# ---------------------------------------------------------------------------
print("Lecture CSV :", CSV_META)
df = pd.read_csv(CSV_META)
"""
def row_to_vec(r):
    x_norm     = r.source_x / 9600
    y_norm     = r.source_y / 9600
    depth_norm = abs(r.source_z) / 9000          # z est négatif
    strike_norm= (r.strike % 360) / 360
    dip_norm   = r.dip / 90
    rake_norm  = ((r.rake + 180) % 360) / 360    # centre sur 0-360
    vs_norm    = 0.                              # ou r.Vs_mean / 4500 si dispo
    depth_dup  = depth_norm                      # 8ᵉ entrée libre
    return [x_norm, y_norm, depth_norm,
            strike_norm, dip_norm, rake_norm,
            vs_norm, depth_dup]

vecs = df.apply(row_to_vec, axis=1).tolist()

text_tensor = torch.tensor(vecs, dtype=torch.float32, device=device)
print("Tensor texte :", text_tensor.shape)
"""
"""
def row_to_vec(r):
    # 5 zéros  pour p/s/coda   (positions 0-4 puis 7)
    v = [0., 0., 0., 0., 0.]
    # distance & azimut
    dx, dy = r.source_x - 4800, r.source_y - 4800
    dz     = r.source_z                      # négatif
    dist_km = (dx*dx + dy*dy + dz*dz) ** 0.5 / 1000
    azim    = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    v += [dist_km / 1, azim / 360]          # normalisés
    v.append(0.)                            # coda_end_sample
    return v
"""


def row_to_vec(r,
               V_P: float = 3000.0,
               V_S: float = 2300.0,
               fs: int  = 100):
    """
    r : Series pandas avec au moins
        - r.trace_name  (ex. 'sample100048_x23_y21')
        - r.source_x, r.source_y, r.source_z
    V_P, V_S : vitesses P/S moyennes (m/s)
    fs       : fréquence d'échantillonnage (Hz)
    """

    # --- 1. indices capteur x,y -----------------------------------------------
    name  = r.trace_name
    parts = name.split('_')
    x_idx = int(parts[1][1:])   # 'x23' -> 23
    y_idx = int(parts[2][1:])   # 'y21' -> 21

    # --- 2. coordonnées capteur [m] ------------------------------------------
    x_sta = 150 + x_idx * 300
    y_sta = 150 + y_idx * 300
    z_sta = 0

    # --- 3. vecteur source→station -------------------------------------------
    dx = r.source_x - x_sta
    dy = r.source_y - y_sta
    dz = r.source_z - z_sta
    dist_m = math.sqrt(dx*dx + dy*dy + dz*dz)

    # --- 4. temps d’arrivée P et S (secondes & échantillons) ---------------
    tP = dist_m / V_P       # en s
    tS = dist_m / V_S       # en s
    sP = tP * fs            # en samples
    sS = tS * fs

    # --- 5. normalisations STEAD --------------------------------------------
    # p_arrival_sample    : /6000
    # p_weight            : on met 1.0 (confiance max)
    # p_travel_sec        : tP/60
    # s_arrival_sample    : /6000
    # s_weight            : 1.0
    # source_distance_km  : dist_m/1000 /300
    # back_azimuth_deg    : (dx,dy) → /360
    # coda_end_sample     : 0

    # back-azimut
    azim_deg = (math.degrees(math.atan2(dx, dy)) + 360) % 360

    v = [
        sP    / 6000.0,       # p_arrival_sample
        0.5,                  # p_weight
        tP    /   60.0,       # p_travel_sec
        sS    / 6000.0,       # s_arrival_sample
        0.5,                  # s_weight
        dist_m/1000.0 / 200,# source_distance_km
        azim_deg / 360.0,     # back_azimuth_deg
        0.13                   # coda_end_sample
    ]
    return v

vecs = np.stack([row_to_vec(r) for _, r in df.iterrows()])   # (N,8)
text_tensor = torch.tensor(vecs, dtype=torch.float32, device=device)
print("Tensor texte :", text_tensor.shape)

# ---------------------------------------------------------------------------
#                         3) encodage & sauvegarde
# ---------------------------------------------------------------------------
with torch.no_grad():
    emb_text = model.encode_text(text_tensor)              # (N, 384)
    emb_text = emb_text / emb_text.norm(dim=-1, keepdim=True)

np.save(OUT_NPY, emb_text.cpu().numpy())
print(f"✓ embeddings texte enregistrés → {OUT_NPY}")

