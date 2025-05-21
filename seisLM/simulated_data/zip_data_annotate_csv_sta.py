#!/home/noam/anaconda3/envs/seislm/bin/python
# zip_data_annotate_csv_sta.py
"""
Batch génère un CSV unique metadata annoté P & S pour tous les HDF5
contenus dans les archives .zip du dossier 'velocities', sans extraction sur disque.
Ajoute une colonne 'split' pour marquer 70% des lignes en 'train' et 30% en 'test'.
"""
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import io
import numpy as np
import pandas as pd
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import hilbert

def compute_p_s_picks_from_bytes(data_bytes: bytes,
                                 station=(0, 0),
                                 dt=0.01,
                                 sta_s=0.05,
                                 lta_s=0.5,
                                 thr=3.0,
                                 min_sep_s=0.5):
    """
    Calcule picks P et S à partir du contenu d'un fichier HDF5 en mémoire.
    """
    import h5py
    with h5py.File(io.BytesIO(data_bytes), 'r') as f:
        uZ = f['uZ'][()]
        uE = f['uE'][()]
        uN = f['uN'][()]
    ix, iy = station
    sig_z = uZ[ix, iy, :].astype(float)
    sig_e = uE[ix, iy, :].astype(float)
    sig_n = uN[ix, iy, :].astype(float)
    h = np.hypot(sig_e, sig_n)
    fs = 1.0 / dt
    # Filtrage passe-bande
    for arr in (sig_z, sig_e, sig_n, h):
        arr[:] = bandpass(arr, 0.5, 20.0, fs, corners=4, zerophase=True)
    # Enveloppes
    env_z = np.abs(hilbert(sig_z))
    env_h = np.abs(hilbert(h))
    # STA/LTA
    nsta = max(1, int(sta_s * fs))
    nlta = min(int(lta_s * fs), env_z.size // 2)
    cft_z = classic_sta_lta(env_z, nsta, nlta)
    cft_h = classic_sta_lta(env_h, nsta, nlta)
    # Pick P
    idxs_p = np.where(cft_z > thr)[0]
    p_idx = int(idxs_p[0]) if idxs_p.size else None
    # Pick S après délai minimal
    min_sep = int(min_sep_s * fs)
    idxs_s = np.where(cft_h > thr)[0]
    if p_idx is not None:
        idxs_s = idxs_s[idxs_s > p_idx + min_sep]
    s_idx = int(idxs_s[0]) if idxs_s.size else None
    return p_idx, s_idx


def main():
    # Répertoires
    base_dir = Path(__file__).parent.resolve()
    vel_dir = base_dir / 'velocities'
    out_csv = base_dir / 'csvs/metadata_annotated_sta.csv'

    print(f"Scanning folder: {vel_dir}")
    zip_paths = sorted(vel_dir.glob('velocity*.zip'))
    if not zip_paths:
        print("No velocity*.zip found. Exiting.")
        return

    # Liste des lignes
    rows = []
    # Colonnes du CSV (ajout de 'split')
    cols = [
        'source_id','source_origin_time','source_origin_uncertainty_sec',
        'source_latitude_deg','source_latitude_uncertainty_km',
        'source_longitude_deg','source_longitude_uncertainty_km',
        'source_depth_km','source_depth_uncertainty_km',
        'source_magnitude','source_magnitude_uncertainty','source_magnitude_type',
        'source_magnitude_author','path_back_azimuth_deg',
        'station_network_code','station_code','trace_channel',
        'station_location_code','station_latitude_deg',
        'station_longitude_deg','station_elevation_m','trace_name',
        'trace_sampling_rate_hz','trace_completeness','trace_has_spikes',
        'trace_start_time','trace_S1_arrival_sample',
        'trace_S1_status','trace_S1_polarity','trace_name_original',
        'trace_P1_arrival_sample','trace_P1_status','trace_P1_polarity',
        'split'
    ]

    # Métadonnées statiques par défaut
    meta_base = {
        'source_origin_time':   datetime.now(timezone.utc).isoformat(),
        'source_origin_uncertainty_sec': '',
        'source_latitude_deg': '',
        'source_latitude_uncertainty_km': '',
        'source_longitude_deg': '',
        'source_longitude_uncertainty_km': '',
        'source_depth_km': '',
        'source_depth_uncertainty_km': '',
        'source_magnitude': '',
        'source_magnitude_uncertainty': '',
        'source_magnitude_type': '',
        'source_magnitude_author': '',
        'path_back_azimuth_deg': '',
        'station_network_code': 'XX',
        'station_code': 'STAT',
        'trace_channel': 'BHZ',
        'station_location_code': '',
        'station_latitude_deg': '',
        'station_longitude_deg': '',
        'station_elevation_m': '',
        'trace_sampling_rate_hz': 100.0,
        'trace_completeness': 1.0,
        'trace_has_spikes': False,
        'trace_start_time': datetime.now(timezone.utc).isoformat()
    }

    # Parcours des archives et des entrées .h5
    for zip_path in zip_paths:
        print(f"Processing {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for entry in zf.namelist():
                if entry.endswith('.h5'):
                    sample_id = Path(entry).stem
                    data_bytes = zf.read(entry)
                    p_idx, s_idx = compute_p_s_picks_from_bytes(data_bytes)
                    # Construire la ligne
                    row = {c: '' for c in cols}
                    row.update(meta_base)
                    row['source_id'] = sample_id
                    row['trace_name'] = sample_id
                    row['trace_P1_arrival_sample'] = p_idx or ''
                    row['trace_P1_status'] = 'AUTO' if p_idx is not None else ''
                    row['trace_S1_arrival_sample'] = s_idx or ''
                    row['trace_S1_status'] = 'AUTO' if s_idx is not None else ''
                    rows.append(row)

    # Écrire tout en un seul CSV
    df = pd.DataFrame(rows, columns=cols)
    # Définir le split: 70% train, 30% test
    n = len(df)
    n_train = int(0.7 * n)
    df['split'] = ['train' if i < n_train else 'test' for i in range(n)]

    df.to_csv(out_csv, index=False)
    print(f"All done. Wrote consolidated CSV: {out_csv}")
    database_csv = base_dir.parent / "inference/simulated_data/metadata.csv"
    df.to_csv(database_csv, index=False)
    print("Also wrote it in database folder:", database_csv)

if __name__ == '__main__':
    main()
