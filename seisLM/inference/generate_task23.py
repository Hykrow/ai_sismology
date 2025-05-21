# generate_task23_csv.py
"""
Generate **task23.csv** from simulated data en reproduisant la logique exacte du
script ETHZ : la longueur globale du signal est déterminée en ouvrant le
fichier **waveforms.hdf5** (structure en « buckets/bucket_i/trace_j », par
exemple *bucket0/onde1*, etc.).

Hypothèse : la colonne `trace_name` du CSV contient le chemin HDF5 complet du
jeu de données (par ex. « bucket0/onde1 »).  Si tel n’est pas le cas, adapte
le code dans la section « # Détermination de global_length » pour reconstruire
le chemin correct (en combinant peut‑être un champ *bucket* + un champ
*trace_id*).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Parameters (identiques au script ETHZ)
# --------------------------------------------------------------------------------------
WINDOW_LENGTH = 1000          # Window length in target samples (at TARGET_SAMPLING_RATE)
TARGET_SAMPLING_RATE = 100    # Desired sampling rate for the CSV (Hz)

# Input paths
METADATA_CSV = Path("simulated_data/metadata.csv")
WAVEFORMS_HDF5 = Path("simulated_data/waveforms.hdf5")

# Output directory and CSV path
SAVE_DIR = Path("simulated_data/targets")
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
# Load metadata & determine global_length via HDF5 (comme le script ETHZ)
# --------------------------------------------------------------------------------------
print("Loading metadata …")

df_meta = pd.read_csv(METADATA_CSV)
# Limit processing to the first 500 traces (copie du script d’origine)
# Supprime la ligne suivante si tu veux traiter l’intégralité des traces
df_meta = df_meta.iloc[:500].reset_index(drop=True)

print("Opening waveforms.hdf5 to determine global trace length …")
with h5py.File(WAVEFORMS_HDF5, "r") as f:
    # On tente d’utiliser le chemin indiqué dans trace_name
    first_path = str(df_meta.loc[0, "trace_name"])
    if first_path in f:
        waveform = f[first_path][()]
    else:
        # Si trace_name ne pointe pas directement sur un dataset, on prend le
        # premier dataset rencontré dans le premier bucket (fallback).
        first_bucket = next(iter(f.keys()))
        first_dataset = next(iter(f[first_bucket].keys()))
        waveform = f[f"{first_bucket}/{first_dataset}"][()]
        print(
            f"⚠️  '{first_path}' introuvable dans HDF5. Fallback sur '{first_bucket}/{first_dataset}'."
        )

    # ETHZ stocke (channels, samples).  Si layout différent, adapte ici.
    global_length = waveform.shape[1] if waveform.ndim == 2 else waveform.shape[0]

print(f"Assuming global length {global_length} samples for all traces (from HDF5).")

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
        row.trace_P1_arrival_sample if not pd.isna(row.trace_P1_arrival_sample) else row.trace_Pg_arrival_sample
    )
    s_pick_global = (
        row.trace_S1_arrival_sample if not pd.isna(row.trace_S1_arrival_sample) else row.trace_Sg_arrival_sample
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
