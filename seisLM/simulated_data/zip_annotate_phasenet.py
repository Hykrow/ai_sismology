#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zip_data_annotate_phasenet_csv.py — v7 (2025‑05‑14)
──────────────────────────────────────────────────
Correctif : le padding est désormais appliqué **à gauche** plutôt qu’à droite
quand la trace est plus courte que `MIN_LEN_S`.

* La fenêtre d’inférence ne tient compte que de l’intervalle contenant les
  données réelles — entre `t_offset` (début des vraies données) et
  `t_end_real` — afin d’exclure totalement la zone de padding gauche.
* Les indices de picks renvoyés (colonnes `trace_P1_arrival_sample` et
  `trace_S1_arrival_sample`) sont exprimés dans le référentiel **de la trace
  originale** (sans padding). Ainsi, l’alignement visuel des traits verticaux
  avec l’onde reste correct dans l’option `--display`.

Interface inchangée :
```bash
python zip_data_annotate_phasenet_csv.py --display 3
```
produira des picks parfaitement alignés.
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, Stream, UTCDateTime
from obspy.signal.filter import bandpass
from seisbench.models import PhaseNet

# ─────────────────────────── Parameters ────────────────────────────
DT: float = 0.01
FS: float = 1.0 / DT
if FS <= 0:
    raise ValueError("Sampling rate FS must be > 0 – check DT constant.")

MIN_LEN_S: float = 60.0
MIN_LEN_SMP: int = int(MIN_LEN_S * FS)

P_FMIN, P_FMAX = 0.5, 20.0
MIN_PROB: float = 0.2

CSV_COLS = [
    'source_id','source_origin_time','source_origin_uncertainty_sec',
    'source_latitude_deg','source_latitude_uncertainty_km','source_longitude_deg',
    'source_longitude_uncertainty_km','source_depth_km','source_depth_uncertainty_km',
    'split','source_magnitude','source_magnitude_uncertainty','source_magnitude_type',
    'source_magnitude_author','path_back_azimuth_deg','station_network_code',
    'station_code','trace_channel','station_location_code','station_latitude_deg',
    'station_longitude_deg','station_elevation_m','trace_name','trace_sampling_rate_hz',
    'trace_completeness','trace_has_spikes','trace_start_time','trace_S1_arrival_sample',
    'trace_S1_status','trace_S1_polarity','trace_name_original','trace_P1_arrival_sample',
    'trace_P1_status','trace_P1_polarity'
]

META_STATIC = {
    'source_origin_time':   datetime.now(timezone.utc).isoformat(),
    'source_origin_uncertainty_sec': '',
    'source_latitude_deg': '',   'source_latitude_uncertainty_km': '',
    'source_longitude_deg': '',  'source_longitude_uncertainty_km': '',
    'source_depth_km': '',       'source_depth_uncertainty_km': '',
    'split': '',
    'source_magnitude': '',      'source_magnitude_uncertainty': '',
    'source_magnitude_type': '', 'source_magnitude_author': '',
    'path_back_azimuth_deg': '',
    'station_network_code': 'XX', 'station_code': 'STAT',
    'trace_channel': 'BHZ',
    'station_location_code': '',
    'station_latitude_deg': '',  'station_longitude_deg': '',
    'station_elevation_m': '',
    'trace_sampling_rate_hz': FS,
    'trace_completeness': 1.0,
    'trace_has_spikes': False,
    'trace_start_time': datetime.now(timezone.utc).isoformat(),
    'trace_name_original': ''
}

# ─────────────────────── Helper functions ─────────────────────────

def _safe_bandpass(sig: np.ndarray) -> np.ndarray:
    try:
        return bandpass(sig, P_FMIN, P_FMAX, FS, corners=4, zerophase=True)
    except Exception as exc:
        logging.warning("bandpass failed (%s); raw used", exc); return sig


def _row(sid: str, p_idx: Optional[int], s_idx: Optional[int]) -> dict:
    r = {c:'' for c in CSV_COLS}; r.update(META_STATIC)
    r['source_id']=sid; r['trace_name']=sid
    if p_idx is not None:
        r['trace_P1_arrival_sample']=p_idx; r['trace_P1_status']='AUTO'
    if s_idx is not None:
        r['trace_S1_arrival_sample']=s_idx; r['trace_S1_status']='AUTO'
    return r

# ─────────────────── PhaseNet inference ─────────────────────

def _infer(data: bytes, model: PhaseNet, want_prob: bool = False):
    """Infer PhaseNet picks *without resampling* (signals are already 100 Hz).
    Le padding est appliqué à **gauche** si nécessaire et totalement exclu de
    la fenêtre de recherche des picks.
    """
    try:
        with h5py.File(io.BytesIO(data), 'r') as f:
            uZ, uE, uN = (np.asarray(f[k]) for k in ('uZ', 'uE', 'uN'))
    except Exception as exc:
        logging.error('Bad HDF5 (%s)', exc)
        return None

    # --- extract 3‑C, scale ----------------------------------------------------
    z = uZ[0, 0].astype('float32', copy=False) * 1000.0  # ×1000 comme la réference
    orig_len = z.size
    e = np.zeros_like(z, dtype='float32')  # horizontales mutées (comportement ref.)
    n = np.zeros_like(z, dtype='float32')

    # --- left padding to reach MIN_LEN_SMP ------------------------------------
    left_pad = max(0, MIN_LEN_SMP - orig_len)
    if left_pad:
        z = np.pad(z, (left_pad, 0))
        e = np.pad(e, (left_pad, 0))
        n = np.pad(n, (left_pad, 0))
    total_len = z.size  # == max(orig_len, MIN_LEN_SMP)

    # Time interval containing *real* data
    t_offset = left_pad / FS                    # start of actual signal (s)
    t_end_real = t_offset + orig_len / FS       # end of actual signal (s)

    # --- build stream ---------------------------------------------------------
    base_start = UTCDateTime()  # common artificial start
    st = Stream([
        Trace(z, header={'channel': 'BHZ', 'sampling_rate': FS, 'starttime': base_start}),
        Trace(n, header={'channel': 'BHN', 'sampling_rate': FS, 'starttime': base_start}),
        Trace(e, header={'channel': 'BHE', 'sampling_rate': FS, 'starttime': base_start}),
    ])

    # --- PhaseNet -------------------------------------------------------------
    try:
        probs = model.annotate(st, strict=False)
    except Exception as exc:
        logging.warning('PhaseNet failed (%s)', exc)
        return None if not want_prob else (None, None, None, None)

    # --- pick inside real window only ----------------------------------------
    best_p = {'v': 0.0, 't': None}; best_s = {'v': 0.0, 't': None}
    for tr in probs:
        fs = tr.stats.sampling_rate
        offset = tr.stats.starttime - base_start
        times = offset + np.arange(tr.data.size) / fs
        mask = (times >= t_offset) & (times <= t_end_real) & (tr.data > 0)
        if not np.any(mask):
            continue
        vals = tr.data[mask]; ts = times[mask]
        idx = int(np.argmax(vals)); val = float(vals[idx]); tt = float(ts[idx])
        if tr.stats.channel.endswith('P') and val > best_p['v']:
            best_p.update(v=val, t=tt)
        elif tr.stats.channel.endswith('S') and val > best_s['v']:
            best_s.update(v=val, t=tt)

    # Convert times to indices within *original* signal (no padding)
    def _time_to_idx(tt: Optional[float]) -> Optional[int]:
        if tt is None:
            return None
        idx = int(round((tt - t_offset) * FS))
        return idx if 0 <= idx < orig_len else None

    p_idx = _time_to_idx(best_p['t'])
    s_idx = _time_to_idx(best_s['t'] if best_s['v'] >= MIN_PROB else None)

    if want_prob:
        # Limit signal to the real part for display
        return p_idx, s_idx, probs, z[left_pad:left_pad+orig_len]
    return p_idx, s_idx, None, None

# ───────────────── Plot helper ─────────────────

def _plot(sid:str,sig:np.ndarray,probs:List[Trace]|None,p_idx:Optional[int],s_idx:Optional[int]):
    if sig is None or probs is None:
        return
    t=np.arange(sig.size)/FS
    fig,ax1=plt.subplots(figsize=(12,4)); ax1.plot(t,sig,'k',label='BHZ (filt)')
    ax2=ax1.twinx(); ax2.set_ylabel('Prob')
    for tr in probs:
        fs=tr.stats.sampling_rate; off=tr.stats.starttime-probs[0].stats.starttime
        tt=off+np.arange(tr.data.size)/fs; mask=(tt>=0)&(tt<t[-1])&(tr.data>0)
        if not np.any(mask): continue
        style,lbl=('b','P prob.') if tr.stats.channel.endswith('P') else ('r--','S prob.')
        ax2.plot(tt[mask],tr.data[mask],style,alpha=0.6,label=lbl if lbl not in ax2.get_legend_handles_labels()[1] else '')
    if p_idx is not None: ax1.axvline(p_idx/FS,color='blue',ls=':',lw=1.5,label=f'P {p_idx/FS:.2f}s')
    if s_idx is not None: ax1.axvline(s_idx/FS,color='red',ls=':',lw=1.5,label=f'S {s_idx/FS:.2f}s')
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc='upper right'); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amp')
    ax1.set_title(sid); plt.tight_layout(); plt.show()

# ─────────────────── Main ──────────────────

def main():
    ap=argparse.ArgumentParser(description='PhaseNet picker (left‑padding safe)')
    ap.add_argument('-v','--vel-dir',type=Path,default=Path(__file__).parent/'velocities')
    ap.add_argument('-d','--device',default='cpu')
    ap.add_argument('-o','--output',type=Path)
    ap.add_argument('--display',type=int,default=0,metavar='N')
    ap.add_argument('-q','--quiet',action='store_true'); args=ap.parse_args()

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING,format='%(levelname)s: %(message)s')
    vel_dir=args.vel_dir.resolve(); out_csv=args.output or vel_dir.parent/"csvs"/'metadata_annotated_phasenet.csv'
    zips=sorted(vel_dir.glob('velocity*.zip'))
    if not zips:
        logging.error('No zip'); return
    model=PhaseNet.from_pretrained('original').to(args.device).eval()

    rows=[]; displays=[]
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for e in zf.namelist():
                if not e.endswith('.h5'): continue
                sid=Path(e).stem; data=zf.read(e)
                want_prob=len(displays)<args.display
                res=_infer(data,model,want_prob)
                if res is None: continue
                p_idx,s_idx,probs,sig=res
                rows.append(_row(sid,p_idx,s_idx))
                if want_prob: displays.append((sid,sig,probs,p_idx,s_idx))
    pd.DataFrame(rows,columns=CSV_COLS).to_csv(out_csv,index=False); logging.info('CSV → %s',out_csv)
    for d in displays: _plot(*d)

if __name__=='__main__':
    main()
