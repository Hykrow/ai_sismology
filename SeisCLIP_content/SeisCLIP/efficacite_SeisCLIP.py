import numpy as np, pandas as pd, torch
from pathlib import Path
import re
BATCH     = 2048          # ajuste selon RAM / GPU
import random

# 1)  chemins -----------------------------------------------------------------
#EMB_AUDIO = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/velocity_emb_with SeisCLIP_384.npy")
#EMB_TEXT  = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/text_emb_384.npy")
#META_CSV  = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data/my_velocity_metadata.csv")
#TRACE_LST = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/trace_names.txt")   # liste ordre audio

EMB_AUDIO = Path("velocity_emb_with SeisCLIP_384.npy")
EMB_TEXT  = Path("text_emb_384.npy")
META_CSV  = Path("MIFNO/data/my_velocity_metadata.csv")
TRACE_LST = Path("trace_names.txt")   # liste ordre audio

# 2)  # chargement & normalisation --------------------------------------------------
emb_audio = torch.tensor(np.load(EMB_AUDIO), dtype=torch.float32)   # (N,384)
emb_text  = torch.tensor(np.load(EMB_TEXT),  dtype=torch.float32)   # (N,384)
emb_audio = emb_audio / emb_audio.norm(dim=-1, keepdim=True)
emb_text  = emb_text  / emb_text.norm(dim=-1, keepdim=True)

trace_names = [l.strip() for l in open(TRACE_LST)]
meta = pd.read_csv(META_CSV).set_index("trace_name")

print(emb_audio.shape[0])
print(emb_text.shape[0])

assert len(trace_names) == emb_audio.shape[0] == emb_text.shape[0]

# =============== TEST DE PROXIMITÉ ==========================
'''import random

coords = meta.loc[trace_names, ['source_x','source_y','source_z']].values
close_pairs, far_pairs = [], []
while len(close_pairs) < 200 and len(far_pairs) < 200:
    i, j = random.sample(range(len(coords)), 2)
    dist = np.linalg.norm(coords[i] - coords[j])
    if dist < 600 and len(close_pairs) < 200:        # proches (<600 m)
        close_pairs.append((i, j))
    elif dist > 5000 and len(far_pairs) < 200:       # lointains (>5 km)
        far_pairs.append((i, j))

cos = torch.nn.functional.cosine_similarity
close_sim = [cos(emb_audio[i], emb_audio[j], dim=0).item()
             for i, j in close_pairs]
far_sim   = [cos(emb_audio[i], emb_audio[j], dim=0).item()
             for i, j in far_pairs]
             
print("\n—— Test proximité audio ——")
print(f"  µ cos proches  = {np.mean(close_sim):.3f}")
print(f"  µ cos lointains= {np.mean(far_sim):.3f}\n")'''

k = 5000
hits = [] 

N = emb_audio.shape[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_text = emb_text.to(device)

top1_idx = np.empty(N, dtype=np.int32)

for start in range(0, N, BATCH):
    end   = min(start + BATCH, N)
    batch = emb_audio[start:end].to(device)          # (B,384)

    with torch.no_grad():
        # similarités (B,384) @ (384,N) -> (B,N)   mais on garde juste le max
        scores      = batch @ emb_text.T            # (B,N)
        top1_idx[start:end] = scores.argmax(dim=1).cpu().numpy()
        topk_idx = scores.topk(k, dim=1).indices.cpu()   # (B,k)
        
# vérité terrain = même indice que l’audio
    gt = torch.arange(start, end).unsqueeze(1)           # (B,1)
    hits.append((topk_idx == gt).any(dim=1).numpy())     # (B,)
    
    del batch, scores  # libère la mémoire GPU



print("similarités calculées par blocs")
"""
#ancienne version qui calcul les coordonnées x y z
# 4)  parser → tableaux (x,y,z,strike,dip,rake)  ------------------------------
true_xyz   = []
true_ang   = []
pred_xyz   = []
pred_ang   = []

for i, name in enumerate(trace_names):
    gt = meta.loc[name]
    pred_name = trace_names[top1_idx[i]]   # description correspondante

    # ground-truth
    true_xyz.append([gt.source_x, gt.source_y, gt.source_z])
    true_ang.append([gt.strike, gt.dip, gt.rake])

    # prédiction  → récupérer directement dans meta (même format)
    pr  = meta.loc[pred_name]
    pred_xyz.append([pr.source_x, pr.source_y, pr.source_z])
    pred_ang.append([pr.strike, pr.dip, pr.rake])

true_xyz  = np.array(true_xyz,  float)
pred_xyz  = np.array(pred_xyz,  float)
true_ang  = np.array(true_ang,  float)
pred_ang  = np.array(pred_ang,  float)

# 5-a)  localisation : distance 3-D ------------------------------------------
eucl_err = np.linalg.norm(pred_xyz - true_xyz, axis=1)    # m
print("── Localisation ─────────────────────────────────────────")
print(f"MAE distance : {eucl_err.mean():.1f} m")
print(f"90ᵉ percentile : {np.percentile(eucl_err,90):.1f} m")
"""
#Nouvelle version qui ne calcul plus que la distance à la source : 

def station_xyz(trace_name):
    """
    trace_name : 'sample100048_x23_y21'
    retourne   : (x_sta, y_sta, z_sta) en mètres
    """
    parts = trace_name.split('_')
    x_idx = int(parts[1][1:])          # 'x23' -> 23
    y_idx = int(parts[2][1:])          # 'y21' -> 21
    x_sta = 150 + x_idx * 300          # gamme 150 … 9 450
    y_sta = 150 + y_idx * 300
    z_sta = 0                          # surface
    return x_sta, y_sta, z_sta

true_d, pred_d = [], []          # listes des distances (km)

for i, name in enumerate(trace_names):
    # capteur de la trace courante
    xs, ys, zs = station_xyz(name)

    # ------------------------------------------------ vérité terrain
    gt = meta.loc[name]                          # ligne ground truth
    dist_true = ((gt.source_x - xs)**2 +
                 (gt.source_y - ys)**2 +
                 (gt.source_z - zs)**2) ** 0.5 / 1000     # km
    true_d.append(dist_true)

    # ------------------------------------------------ prédiction
    pred_name = trace_names[top1_idx[i]]         # texte le plus proche
    pr  = meta.loc[pred_name]
    dist_pred = ((pr.source_x - xs)**2 +
                 (pr.source_y - ys)**2 +
                 (pr.source_z - zs)**2) ** 0.5 / 1000     # km
    pred_d.append(dist_pred)

true_d  = np.array(true_d)
pred_d  = np.array(pred_d)
err_km  = np.abs(pred_d - true_d)                # écart absolu en km

print("── Distance source-station ──────────────────────────────")
print(f"MAE distance : {err_km.mean()*1000:.1f} m")
for pct in (50, 75, 90, 95, 99):
    print(f"{pct:>3}ᵉ percentile : {np.percentile(err_km, pct)*1000:.1f} m")

"""
#Ancienne version qui calcul l'erreur angulaire
# 5-b)  mécanisme : erreur angulaire mod 360 -------------------
circ = lambda p,t: np.abs(((p-t+180)%360)-180)
strike_err = circ(pred_ang[:,0], true_ang[:,0])
dip_err    = circ(pred_ang[:,1], true_ang[:,1])
rake_err   = circ(pred_ang[:,2], true_ang[:,2])

print("── Angles (°) ──────────────────────────────────────────")
for name, err in zip(("strike","dip","rake"),
                     (strike_err,dip_err,rake_err)):
    print(f"{name:<6} : MAE {err.mean():5.1f}°   90ᵉ pct {np.percentile(err,90):5.1f}°")
"""
# 6)  retrieval Top-k (k=5) -----------------------------------

hits = np.concatenate(hits)
recall = hits.mean() * 100
print(f"── Recall@{k} : {recall:.2f}%")

# Immédiatement après avoir chargé et normalisé :
# emb_audio = torch.tensor(...); emb_text = torch.tensor(...)
emb_audio = emb_audio.cpu()
emb_text  = emb_text.cpu()

# Test 3 : propre vs décalé
own_scores  = (emb_audio[:100] * emb_text[:100]).sum(dim=1)   # tenseurs CPU
other_scores= (emb_audio[:100] * emb_text[1:101]).sum(dim=1)

own   = own_scores.mean().item()
other = other_scores.mean().item()
#print(f"Score propre  : {own:.3f}   |   Score décalé : {other:.3f}")

own_all = (emb_audio * emb_text).sum(dim=1).cpu().numpy()
#print(own_all.min(), own_all.mean(), own_all.max())
# simple décalage circulaire
rand_all = (emb_audio * emb_text.roll(shifts=1, dims=0)).sum(dim=1).cpu().numpy()


#print(emb_audio.norm(dim=1).mean(), emb_text.norm(dim=1).mean())

import matplotlib.pyplot as plt

'''
plt.hist(own_all, bins=50, alpha=0.7, label="own (i,i)")
plt.hist(rand_all, bins=50, alpha=0.7, label="random (i,i+1)")
plt.xlabel("Cosinus similarity")
plt.ylabel("Nombre de traces")
plt.legend()
plt.title("Distribution own vs random")
plt.show()
'''

#dist_err = eucl_err  # numpy array
#for pct in [50, 75, 90, 95, 99]:
#    print(f"{pct}ᵉ percentile : {np.percentile(dist_err, pct):.1f} m")
'''
# CDF plot
sorted_err = np.sort(dist_err)
p = np.linspace(0, 1, len(sorted_err))
plt.plot(sorted_err, p)
plt.xlabel("Erreur de localisation (m)")
plt.ylabel("Fraction ≤ erreur")
plt.title("CDF des erreurs de localisation")
plt.grid(True)
plt.show()
'''
'''
true_xyz = np.array(true_xyz)
centre = np.mean(true_xyz, axis=0)
dists_from_centre = np.linalg.norm(true_xyz - centre, axis=1)

plt.scatter(dists_from_centre, dist_err, s=5, alpha=0.5)
plt.xlabel("Distance source → centre du domaine (m)")
plt.ylabel("Erreur prédiction (m)")
plt.title("Erreur vs position absolue")
plt.grid(True)
plt.show()
'''
'''
# scores_diag[i] = emb_audio[i]·emb_text[i]
scores_diag = (emb_audio * emb_text).sum(dim=1).cpu().numpy()
meta_feats = meta[['source_x','source_y','source_z','strike','dip','rake']].values
corrs = np.corrcoef(meta_feats.T, scores_diag)[-1, :-1]

plt.bar(['x','y','z','strike','dip','rake'], corrs)
plt.ylabel("Corrélation avec cos(own)")
plt.title("Corrélation features → similarity")
plt.grid(True, axis="y")
plt.show()
'''
print("✓ évaluation terminée")
