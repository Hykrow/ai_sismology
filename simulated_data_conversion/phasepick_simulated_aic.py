#!/usr/bin/env python3
# phasepick_simulated_aic_manual.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

# --- 1. Chargement des données ---
with h5py.File("sample106201.h5", "r") as f:
    uE = f["uE"][:]
    uN = f["uN"][:]
    uZ = f["uZ"][:]

# --- 2. Paramètres temporels & station ---
dt, fs = 0.01, 1/0.01
ix, iy = 0, 0
sig_z = uZ[ix, iy, :].astype(float)
sig_e = uE[ix, iy, :].astype(float)
sig_n = uN[ix, iy, :].astype(float)
sig_h = np.hypot(sig_e, sig_n)
npts = sig_z.size
time = np.arange(npts) * dt

# --- 3. Filtrage passe-bande 1–20 Hz ---
sig_z = bandpass(sig_z, 1.0, 20.0, fs, corners=4, zerophase=True)
sig_h = bandpass(sig_h, 1.0, 20.0, fs, corners=4, zerophase=True)

# --- 4. Calcul AIC manuel ---
def compute_aic(trace):
    N = trace.size
    c1 = trace.cumsum()
    c2 = (trace**2).cumsum()
    aic = np.zeros(N, float)
    eps = 1e-10
    for k in range(1, N-1):
        n1, n2 = k, N-k
        v1 = (c2[k] - c1[k]**2/n1) / n1
        v2 = ((c2[-1]-c2[k]) - (c1[-1]-c1[k])**2/n2) / n2
        aic[k] = n1*np.log(v1+eps) + n2*np.log(v2+eps)
    return aic

aic_z = compute_aic(sig_z)
aic_h = compute_aic(sig_h)

# --- 5. Repérage préliminaire par seuil bas (2%) ---
threshold_frac = 0.02
env_z = np.abs(sig_z)
th_z = threshold_frac * env_z.max()
cand_z = np.where(env_z > th_z)[0]
if cand_z.size == 0:
    raise RuntimeError("Aucun candidat P sur Z")
prelim_p = cand_z[0]

# --- 6. Affinage AIC-P ---
pre_win, post_win = int(0.2*fs), int(1.0*fs)
start_p = max(prelim_p - pre_win, 0)
end_p   = min(prelim_p + post_win, npts-1)
seg_aic_z = aic_z[start_p:end_p]
p_rel = np.argmin(seg_aic_z)
p_idx = start_p + p_rel

# --- 7. Repérage préliminaire S après P+0.5s ---
min_sep = int(0.5 * fs)
env_h   = np.abs(sig_h)
th_h    = threshold_frac * env_h.max()
cand_h  = np.where(env_h > th_h)[0]
cand_h  = cand_h[cand_h > p_idx + min_sep]
if cand_h.size == 0:
    raise RuntimeError("Aucun candidat S sur H après P+0.5s")
prelim_s = cand_h[0]

# Affinage AIC-S ---
start_s = max(prelim_s - pre_win, p_idx + min_sep)
end_s   = min(prelim_s + post_win, npts-1)
seg_aic_h = aic_h[start_s:end_s]
s_rel = np.argmin(seg_aic_h)
s_idx = start_s + s_rel

# --- 8. Affichage des résultats ---
plt.figure(figsize=(12, 8))

# Z + P pick only
ax1 = plt.subplot(3,1,1)
ax1.plot(time, sig_z, 'k', label='Z filtrée')
ax1.axvline(p_idx*dt, color='r', ls='--', label=f'P pick @ {p_idx*dt:.2f}s')
ax1.set(title=f"Station ({ix},{iy}) — Composante Z", ylabel="Amp")
ax1.legend()

# E + S pick only
ax2 = plt.subplot(3,1,2)
ax2.plot(time, sig_e, 'b', label='E brut filtré')
ax2.axvline(s_idx*dt, color='g', ls='--', label=f'S pick @ {s_idx*dt:.2f}s')
ax2.set(title="Composante E", ylabel="Amp")
ax2.legend()

# N + S pick only
ax3 = plt.subplot(3,1,3)
ax3.plot(time, sig_n, 'm', label='N brut filtré')
ax3.axvline(s_idx*dt, color='g', ls='--')
ax3.set(title="Composante N", xlabel="Temps (s)", ylabel="Amp")
ax3.legend()

plt.tight_layout()
plt.show()

print(f">>> Pick P final à {p_idx*dt:.2f} s (indice {p_idx})")
print(f">>> Pick S final à {s_idx*dt:.2f} s (indice {s_idx})")