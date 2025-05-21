#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison_sta_vs_preds.py — v10 (2025-05-16)
──────────────────────────────────────────────────
Affiche directement la comparaison STA/LTA ↔ nouveau CSV en ne gardant
que les prédictions dont la probabilité (score_detection) est supérieure à
un seuil passé en paramètre, et affiche des statistiques d'erreur.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────── Constantes ────────────────────────────────

BASE_DIR    = Path(__file__).parent.resolve()
CSV_PATH =BASE_DIR/"csvs"
STA_CSV     = CSV_PATH / "metadata_annotated_sta.csv"
NEW_CSV     = BASE_DIR.parent / "predictions" / "preds.csv"

# Colonnes d’intérêt
COL_P_STA   = "trace_P1_arrival_sample"
COL_S_STA   = "trace_S1_arrival_sample"
RATE        = "trace_sampling_rate_hz"
COL_SCORE   = "score_detection"
COL_P_NEW   = "p_sample_pred"
COL_S_NEW   = "s_sample_pred"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare les picks STA/LTA vs prédictions avec un seuil de probabilité"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.3,
        help="Seuil minimal de score_detection pour filtrer les prédictions"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresh = args.threshold

    # Vérif des fichiers
    if not STA_CSV.is_file() or not NEW_CSV.is_file():
        logging.error("Assurez-vous que %s et %s existent", STA_CSV.name, NEW_CSV.name)
        return

    # Chargement des DataFrames
    sta_df = pd.read_csv(STA_CSV, low_memory=False)
    if 'split' in sta_df.columns:
        sta_df = sta_df[sta_df['split'] == 'test']
    else:
        logging.warning("La colonne 'split' n'existe pas dans %s", STA_CSV.name)

    new_df = pd.read_csv(NEW_CSV, low_memory=False)

    # Renommer la clé de trace si besoin
    if "trace_name" in new_df.columns:
        new_df = new_df.rename(columns={"trace_name": "source_id"})

    # Jointure
    merged = pd.merge(
        sta_df[["source_id", COL_P_STA, COL_S_STA, RATE]],
        new_df[["source_id", COL_SCORE, COL_P_NEW, COL_S_NEW]],
        on="source_id", how="inner"
    )

    # Conversion en secondes
    merged["P_sta_s"] = merged[COL_P_STA] / merged[RATE]
    merged["P_new_s"] = merged[COL_P_NEW] / merged[RATE]
    merged["S_sta_s"] = merged[COL_S_STA] / merged[RATE]
    merged["S_new_s"] = merged[COL_S_NEW] / merged[RATE]

    # Filtre par probabilité
    mask_prob = merged[COL_SCORE] > thresh
    df_p = merged.loc[mask_prob & merged["P_sta_s"].notna() & merged["P_new_s"].notna()]
    df_s = merged.loc[mask_prob & merged["S_sta_s"].notna() & merged["S_new_s"].notna()]

    # Calcul des statistiques d'erreur
    def compute_stats(df: pd.DataFrame, col_new: str, col_sta: str) -> dict[str, float]:
        errs = df[col_new] - df[col_sta]
        return {
            'count': len(errs),
            'mean': errs.mean(),
            'variance': errs.var(),
            'std_dev': errs.std(),
            'rmse': (errs**2).mean()**0.5,
        }

    stats_p = compute_stats(df_p, 'P_new_s', 'P_sta_s')
    stats_s = compute_stats(df_s, 'S_new_s', 'S_sta_s')

    # Affichage des stats
    logging.info("Statistiques pour picks P (score > %0.2f):", thresh)
    logging.info("  count      : %d", stats_p['count'])
    logging.info("  mean error : %0.4f s", stats_p['mean'])
    logging.info("  variance   : %0.4f s²", stats_p['variance'])
    logging.info("  std dev    : %0.4f s", stats_p['std_dev'])
    logging.info("  rmse       : %0.4f s", stats_p['rmse'])

    logging.info("Statistiques pour picks S (score > %0.2f):", thresh)
    logging.info("  count      : %d", stats_s['count'])
    logging.info("  mean error : %0.4f s", stats_s['mean'])
    logging.info("  variance   : %0.4f s²", stats_s['variance'])
    logging.info("  std dev    : %0.4f s", stats_s['std_dev'])
    logging.info("  rmse       : %0.4f s", stats_s['rmse'])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    title_suffix = f"(prob > {thresh})"

    # Picks P
    if not df_p.empty:
        x, y = df_p['P_sta_s'], df_p['P_new_s']
        axes[0].scatter(x, y, s=12, alpha=0.7)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        axes[0].plot(lims, lims, "k--", lw=1)
        axes[0].set(
            xlabel=f"P_sta (s)", ylabel=f"P_new (s)",
            xlim=lims, ylim=lims,
            title=f"Picks P {title_suffix}"
        )
        axes[0].grid(ls=":")
    else:
        axes[0].text(.5, .5, f"Pas de picks P {title_suffix}", ha="center", va="center")
        axes[0].axis("off")

    # Picks S
    if not df_s.empty:
        x, y = df_s['S_sta_s'], df_s['S_new_s']
        axes[1].scatter(x, y, s=12, alpha=0.7, marker="x")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        axes[1].plot(lims, lims, "k--", lw=1)
        axes[1].set(
            xlabel=f"S_sta (s)", ylabel=f"S_new (s)",
            xlim=lims, ylim=lims,
            title=f"Picks S {title_suffix}"
        )
        axes[1].grid(ls=":")
    else:
        axes[1].text(.5, .5, f"Pas de picks S {title_suffix}", ha="center", va="center")
        axes[1].axis("off")

    fig.suptitle(f"STA/LTA vs seisLM — Scatter des picks filtrés {title_suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
