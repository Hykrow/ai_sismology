#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_picks_sta_vs_phasenet.py — v5 (2025-05-14)
──────────────────────────────────────────────────
Affiche **directement** (pas de PNG) la comparaison PhaseNet ↔ STA/LTA :

• Scatter P (abscisse = pick STA/LTA, ordonnée = pick PhaseNet)
• Scatter S (idem) sur le même canvas à deux panneaux.

Les fichiers recherchés :
    velocities/metadata_annotated_all.csv          ← ground‑truth STA/LTA
    velocities/metadata_phasenet_annotated_all.csv ← prédictions PhaseNet

Exécuter sans argument :
    $ python compare_picks_sta_vs_phasenet.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────── Configuration ───────────────────────────────
BASE_DIR  = Path(__file__).parent.resolve()
VEL_DIR   = BASE_DIR / "velocities"
STA_CSV   = VEL_DIR / "metadata_annotated_all.csv"
PN_CSV    = VEL_DIR / "metadata_phasenet_annotated_all.csv"

COL_P = "trace_P1_arrival_sample"
COL_S = "trace_S1_arrival_sample"
RATE  = "trace_sampling_rate_hz"  # 100 Hz nominal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ───────────────────────────────── Main ────────────────────────────────────

def main() -> None:
    # Vérification des CSV
    if not STA_CSV.is_file() or not PN_CSV.is_file():
        logging.error("Assurez‑vous que %s et %s existent", STA_CSV.name, PN_CSV.name)
        return

    # Chargement
    sta_df = pd.read_csv(STA_CSV, low_memory=False)
    pn_df  = pd.read_csv(PN_CSV,  low_memory=False)

    # Jointure sur source_id
    merged = pd.merge(
        sta_df[["source_id", COL_P, COL_S, RATE]],
        pn_df [["source_id", COL_P, COL_S]],
        on="source_id", how="inner", suffixes=("_sta", "_pn"),
    )

    # Conversion en secondes
    merged["P_sta_s"] = merged[f"{COL_P}_sta"] / merged[RATE]
    merged["P_pn_s"]  = merged[f"{COL_P}_pn"]  / merged[RATE]
    merged["S_sta_s"] = merged[f"{COL_S}_sta"] / merged[RATE]
    merged["S_pn_s"]  = merged[f"{COL_S}_pn"]  / merged[RATE]

    mask_p = merged["P_sta_s"].notna() & merged["P_pn_s"].notna()
    mask_s = merged["S_sta_s"].notna() & merged["S_pn_s"].notna()

    # Figure — deux scatters côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)

    if mask_p.any():
        ax = axes[0]
        x, y = merged.loc[mask_p, "P_sta_s"], merged.loc[mask_p, "P_pn_s"]
        ax.scatter(x, y, s=12, alpha=0.7)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("P_sta (s)"); ax.set_ylabel("P_pn (s)")
        ax.set_title("Comparaison des picks P")
        ax.grid(ls=":")
    else:
        axes[0].text(0.5, 0.5, "Pas de picks P communs", ha="center", va="center")
        axes[0].axis("off")

    if mask_s.any():
        ax = axes[1]
        x, y = merged.loc[mask_s, "S_sta_s"], merged.loc[mask_s, "S_pn_s"]
        ax.scatter(x, y, s=12, alpha=0.7, marker="x")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("S_sta (s)"); ax.set_ylabel("S_pn (s)")
        ax.set_title("Comparaison des picks S")
        ax.grid(ls=":")
    else:
        axes[1].text(0.5, 0.5, "Pas de picks S communs", ha="center", va="center")
        axes[1].axis("off")

    fig.suptitle("PhaseNet vs. STA/LTA — Scatter des picks", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
