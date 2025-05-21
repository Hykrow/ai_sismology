# generate_task23_csv.py
"""
Génère **task23.csv** depuis ton dataset simulé en utilisant directement
`WaveformDataset` de **SeisBench**, exactement comme on fait avec
`ETHZ(cache="full")`.

Avantages :
* Gestion automatique des **buckets** (structure `data/bucket_i/trace_j`).
* Même API que `ETHZ` : `metadata`, `get_sample`, etc.
* Plus besoin de manipuler `h5py` manuellement.

Pré‑requis sur ton dossier `simulated_data/` :
* `metadata.csv` (champ `trace_name` faisant référence au chemin HDF5 relatif
  **à l’intérieur** du groupe `data/` – c’est ce que produit
  `WaveformDataWriter`).
* `waveforms.hdf5` avec le groupe `data/` contenant les traces, buckets ou non.

Si ces conditions sont remplies, le script est interchangeable avec celui basé
sur ETHZ.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ➡️  SeisBench
from seisbench.data import WaveformDataset

# --------------------------------------------------------------------------------------
# Parameters (identiques au script ETHZ)
# --------------------------------------------------------------------------------------
WINDOW_LENGTH = 1000          # Window length in target samples (at TARGET_SAMPLING_RATE)
TARGET_SAMPLING_RATE = 100    # Desired sampling rate for the CSV (Hz)

# Input directory that contains metadata.csv + waveforms.hdf5

# Répertoire où se trouve ce fichier .py
BASE_DIR = Path(__file__).resolve().parent

# Le dossier simulated_data, toujours à côté de votre script
DATA_DIR = BASE_DIR / "simulated_data"

# Output directory and CSV path
SAVE_DIR = DATA_DIR 
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = SAVE_DIR / "task23.csv"

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def compute_local_length(global_length: int, original_sr: float):
    """Convert *global_length* (samples @ original_sr) to local_length (@ TARGET_SAMPLING_RATE)."""
    conversion_factor = TARGET_SAMPLING_RATE / original_sr
    return int(round(global_length * conversion_factor)), conversion_factor


def choose_window(pick_local: float, local_length: int, window_length: int = WINDOW_LENGTH):
    """Return (start, end) of a random window containing *pick_local* (local sample idx)."""
    if np.isnan(pick_local):
        pick_local = local_length / 2

    p_int = int(round(pick_local))
    lower_bound = max(0, p_int - window_length + 1)
    upper_bound = min(p_int, local_length - window_length)

    if lower_bound <= upper_bound:
        start = np.random.randint(lower_bound, upper_bound + 1)
    else:  # pick near an edge – centre the window
        start = max(0, p_int - window_length // 2)
        if start + window_length > local_length:
            start = local_length - window_length

    return start, start + window_length

# --------------------------------------------------------------------------------------
# Load dataset via WaveformDataset (équivalent ETHZ(cache="full"))
# --------------------------------------------------------------------------------------
print("Loading WaveformDataset …")

dataset = WaveformDataset(DATA_DIR, cache="full")

df_meta = dataset.metadata.copy().reset_index(drop=True)
df_meta = df_meta[df_meta['split'] == 'test'].reset_index(drop=True)
print("LONGUEUR TOT", len(df_meta))
# Limiter à 500 traces pour coller au script ETHZ ; supprime la ligne si besoin
#df_meta = df_meta.iloc[:500].reset_index(drop=True)

waveform, _ = dataset.get_sample(0)
# ETHZ stocke (channels, samples). WaveformDataset fait pareil.
global_length = waveform.shape[1]
print(f"Assuming global length {global_length} samples for all traces (from first sample).")

# --------------------------------------------------------------------------------------
# Main processing loop – identique au script ETHZ
# --------------------------------------------------------------------------------------
rows: list[dict[str, float | int | str]] = []
trace_idx_counter = 0

for row in tqdm(df_meta.itertuples(index=False), total=len(df_meta), desc="Processing traces"):
    trace_name = row.trace_name
    trace_split = "test"  # forcé à "test" comme dans l’original

    # Sampling rate of this trace in the original domain (Hz)
    original_sr = row.trace_sampling_rate_hz
    local_length, conversion_factor = compute_local_length(global_length, original_sr)

    # Picks in global domain (original sampling rate)
    p_pick_global = (
        row.trace_P1_arrival_sample if not pd.isna(row.trace_P1_arrival_sample) else 799
    )
    s_pick_global = (
        row.trace_S1_arrival_sample if not pd.isna(row.trace_S1_arrival_sample) else 799
    )

    # -------------------------------------
    # Phase P
    # -------------------------------------
    if not pd.isna(p_pick_global):
        p_pick_local = p_pick_global * conversion_factor
        start_sample, end_sample = choose_window(p_pick_local, local_length)
        rows.append(
            {
                "trace_name": trace_name,
                "trace_idx": trace_idx_counter,
                "trace_split": trace_split,
                "sampling_rate": TARGET_SAMPLING_RATE,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "phase_label": "P",
                "full_phase_label": "P1" if not pd.isna(row.trace_P1_arrival_sample) else "Pg",
                "phase_onset": p_pick_local,
            }
        )

    # -------------------------------------
    # Phase S
    # -------------------------------------
    if not pd.isna(s_pick_global):
        s_pick_local = s_pick_global * conversion_factor
        start_sample, end_sample = choose_window(s_pick_local, local_length)
        rows.append(
            {
                "trace_name": trace_name,
                "trace_idx": trace_idx_counter,
                "trace_split": trace_split,
                "sampling_rate": TARGET_SAMPLING_RATE,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "phase_label": "S",
                "full_phase_label": "S1" if not pd.isna(row.trace_S1_arrival_sample) else "Sg",
                "phase_onset": s_pick_local,
            }
        )

    trace_idx_counter += 1

# --------------------------------------------------------------------------------------
# Save CSV
# --------------------------------------------------------------------------------------
print("Writing CSV …")

df_csv = pd.DataFrame(rows)
df_csv.sort_values(by=["trace_idx", "phase_label"], inplace=True)
df_csv.to_csv(OUTPUT_CSV, index=False)

print("CSV generated:", OUTPUT_CSV)
