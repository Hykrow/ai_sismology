#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_models_stats_extended.py

Trace des barplots de variance, erreur moyenne et count pour différents modèles de pick.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_bars(df: pd.DataFrame, cols: list[str], title: str, ylabel: str) -> None:
    """
    Trace un barplot groupé pour les colonnes `cols` sur chaque modèle (index de df).
    """
    x = np.arange(len(df.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df[cols[0]], width, label=cols[0])
    ax.bar(x + width/2, df[cols[1]], width, label=cols[1])
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(ls=':')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Statistiques extraites de vos logs
    stats = {
        'PhaseNet': {
            'mean_p': 0.1197,   'variance_p': 0.1447,  'count_p': 800,
            'mean_s': 0.6381,   'variance_s': 0.3552,  'count_s': 47,
        },
        'seisLM (no FT)': {
            'mean_p': 2.5796,   'variance_p': 7.7441,  'count_p': 630,
            'mean_s': -0.0690,  'variance_s': 1.3632,  'count_s': 628,
        },
        'seisLM FT données réelles (~30k)': {
            'mean_p': 0.7051,   'variance_p': 0.8391,  'count_p': 690,
            'mean_s': 0.1884,   'variance_s': 1.4613,  'count_s': 688,
        },
        'seisLM FT données simulées (~2k)': {
            'mean_p': 1.0443,   'variance_p': 0.7242,  'count_p': 304,
            'mean_s': 0.2429,   'variance_s': 1.0812,  'count_s': 304,
        },
    }

    # DataFrame
    df = pd.DataFrame(stats).T

    # 1) Variance pour P et S
    plot_bars(
        df,
        cols=['variance_p', 'variance_s'],
        title="Comparaison des variances entre modèles (P vs S)",
        ylabel="Variance (s²)"
    )

    # 2) Erreur moyenne pour P et S
    plot_bars(
        df,
        cols=['mean_p', 'mean_s'],
        title="Comparaison des erreurs moyennes entre modèles (P vs S)",
        ylabel="Erreur moyenne (s)"
    )

    # 3) Count pour P et S
    plot_bars(
        df,
        cols=['count_p', 'count_s'],
        title="Comparaison du nombre de picks (P vs S)",
        ylabel="Count"
    )
