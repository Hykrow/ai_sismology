#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_zips_to_single_hdf5.py — v3.1 (2025‑05‑14)
─────────────────────────────────────────────────
Regroupe les traces .h5 de tous les `velocity*.zip` en **un unique** fichier
Waveforms‑ETHZ. Toutes les traces sont enregistrées directement dans le groupe
`/data` (plus de sous‑groupes *bucket*).

Un seul CSV `metadata_all.csv` récapitule toutes les traces.

Usage :
```bash
python convert_zips_to_single_hdf5.py \
        --vel-dir velocities \
        --output-h5 waveforms_all.hdf5 \
        --metadata-csv metadata_all.csv
```
Options :
  • `--quiet` : logs minimalistes.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

# ──────────────────────────── Constantes ────────────────────────────
COMPONENTS = [('Z', 'uZ'), ('N', 'uN'), ('E', 'uE')]
ETHZ_FMT = {
    'component_order': b'ZNE',
    'dimension_order': b'CW',
    'instrument_response': b'not restituted',
    'measurement': b'velocity',
    'unit': b'counts',
}
CSV_COLUMNS = [
    'source_id','source_origin_time','source_origin_uncertainty_sec',
    'source_latitude_deg','source_latitude_uncertainty_km',
    'source_longitude_deg','source_longitude_uncertainty_km',
    'source_depth_km','source_depth_uncertainty_km','split',
    'source_magnitude','source_magnitude_uncertainty',
    'source_magnitude_type','source_magnitude_author',
    'path_back_azimuth_deg','station_network_code','station_code',
    'trace_channel','station_location_code',
    'station_latitude_deg','station_longitude_deg','station_elevation_m',
    'trace_name','trace_sampling_rate_hz','trace_completeness',
    'trace_has_spikes','trace_start_time',
    'trace_S1_arrival_sample','trace_S1_status','trace_S1_polarity',
    'trace_name_original','trace_P1_arrival_sample','trace_P1_status',
    'trace_P1_polarity','trace_Pg_arrival_sample','trace_Pg_status',
    'trace_Pg_polarity','trace_Sg_arrival_sample','trace_Sg_status',
    'trace_Sg_polarity','trace_SmS_arrival_sample','trace_SmS_status',
    'trace_SmS_polarity','trace_PmP_arrival_sample','trace_PmP_status',
    'trace_PmP_polarity','trace_Pn_arrival_sample','trace_Pn_status',
    'trace_Pn_polarity','trace_P_arrival_sample','trace_P_status',
    'trace_P_polarity','trace_Sn_arrival_sample','trace_Sn_status',
    'trace_Sn_polarity'
]
SAMPLING_RATE = 100.0

# ────────────────────────── Helper functions ─────────────────────────

def read_waveform_from_h5(data: bytes) -> np.ndarray:
    """Lit les datasets 'uZ','uN','uE' et retourne un array (3, N)."""
    with h5py.File(io.BytesIO(data), 'r') as f:
        comps: Dict[str, np.ndarray] = {}
        for comp, name in COMPONENTS:
            if name not in f or not isinstance(f[name], h5py.Dataset):
                raise KeyError(f"Dataset '{name}' manquant")
            arr = f[name][()]
            if arr.ndim == 3:
                arr = arr[0, 0, :]
            comps[comp] = arr.astype(np.float64)
        return build_3xN_array(comps)


def build_3xN_array(comps: Dict[str, np.ndarray]) -> np.ndarray:
    lengths = [comps[c].size for c, _ in COMPONENTS]
    n = max(lengths)
    data = np.zeros((3, n), dtype=np.float64)
    for i, (comp, _) in enumerate(COMPONENTS):
        arr = comps[comp].ravel()
        data[i, : arr.size] = arr
    return data


def create_target_h5(path: Path) -> h5py.File:
    """Crée un fichier HDF5 ETHZ racine (écrasé si présent)."""
    if path.exists():
        path.unlink()
    f = h5py.File(path, 'w')
    fmt = f.create_group('data_format')
    for k, v in ETHZ_FMT.items():
        fmt.create_dataset(k, data=np.string_(v))
    # S'assure que le groupe /data existe
    f.require_group('data')
    return f


def make_metadata_row(trace_id: str, *, source_id: str) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    base = {
        'source_id': source_id,
        'source_origin_time': now_iso,
        'source_origin_uncertainty_sec': '',
        'source_latitude_deg': '', 'source_latitude_uncertainty_km': '',
        'source_longitude_deg': '', 'source_longitude_uncertainty_km': '',
        'source_depth_km': '', 'source_depth_uncertainty_km': '', 'split': '',
        'source_magnitude': '', 'source_magnitude_uncertainty': '',
        'source_magnitude_type': '', 'source_magnitude_author': '',
        'path_back_azimuth_deg': '', 'station_network_code': 'XX',
        'station_code': 'STAT', 'trace_channel': 'BHZ',
        'station_location_code': '', 'station_latitude_deg': '',
        'station_longitude_deg': '', 'station_elevation_m': '',
        'trace_name': trace_id, 'trace_sampling_rate_hz': SAMPLING_RATE,
        'trace_completeness': 1.0, 'trace_has_spikes': False,
        'trace_start_time': now_iso, 'trace_name_original': '',
    }
    return {c: base.get(c, '') for c in CSV_COLUMNS}

# ───────────────────────────── Main ────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Regroupe tous les .h5 des ZIP en un Waveforms‑ETHZ + CSV (sans buckets)")
    ap.add_argument('-v', '--vel-dir', type=Path, default=Path('velocities'))
    ap.add_argument('--output-h5', type=Path, default=Path('/home/noam/seisLM/inference/simulated_data/waveforms.hdf5'))
    ap.add_argument('--metadata-csv', type=Path, default=Path('metadata_all.csv'))
    ap.add_argument('-q', '--quiet', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING,
                        format='%(levelname)s: %(message)s')

    vel_dir = args.vel_dir.resolve()
    if not vel_dir.exists():
        logging.error("Dossier %s introuvable", vel_dir)
        sys.exit(1)

    h5_path = args.output_h5.resolve()
    csv_path = args.metadata_csv.resolve()

    h5 = create_target_h5(h5_path)
    data_group = h5['data']

    rows: List[dict] = []
    n_added = 0

    zips = sorted(vel_dir.glob('velocity*.zip'))
    if not zips:
        logging.error("Aucun velocity*.zip trouvé dans %s", vel_dir)
        sys.exit(1)

    for z in zips:
        logging.info("Traitement %s", z.name)
        with zipfile.ZipFile(z) as zf:
            for fname in sorted(zf.namelist()):
                if not fname.endswith('.h5'):
                    continue
                try:
                    waveform = read_waveform_from_h5(zf.read(fname))
                except Exception as exc:
                    logging.warning("%s → ignoré (%s)", fname, exc)
                    continue

                trace_base = Path(fname).stem
                trace_id = trace_base
                if trace_id in data_group:
                    logging.warning("Trace %s déjà présente — ignorée", trace_id)
                    continue

                data_group.create_dataset(trace_id, data=waveform,
                                          compression='gzip', shuffle=True)

                rows.append(make_metadata_row(trace_id, source_id=trace_base))
                n_added += 1
                logging.info("✚ %s → /data (index %d)", trace_id, n_added)

    # Écriture CSV and close h5
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    logging.info("✅ CSV écrit : %s (lignes : %d)", csv_path, len(rows))
    h5.close()
    logging.info("✅ HDF5 finalisé : %s (traces : %d)", h5_path, n_added)


if __name__ == '__main__':
    main()
