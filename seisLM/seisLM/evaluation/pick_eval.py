# -*- coding: utf-8 -*-
"""pick_eval_mod.py — v0.11
────────────────────────────────────────────────────────────────
Instrumentation *maximale* : on imprime tout ce qu’il est possible de voir
– dataset, split, generator, loader, batch, HDF5 groups et mapping.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L

import h5py
import seisbench.data as sbd
import seisbench.generate as sbg
from seisbench.data import WaveformDataset

# ────────────────────────── Logging ──────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True
)
log = logging.getLogger("pick_eval_mod")

DATA_DIR = Path("/home/noam/seisLM/inference/simulated_data")

###############################################################################
# Patch helper pour lookup très verbeux ↴
###############################################################################

def _patch_lookup_verbose(ds_like):
    md = ds_like.metadata
    md.reset_index(drop=True, inplace=True)
    if 'trace_name' not in md.columns and 'name' in md.columns:
        log.warning("Colonne 'trace_name' absente ➜ renomme 'name'")
        md.rename(columns={'name':'trace_name'}, inplace=True)

    mapping: Dict[str,int] = {nm: i for i, nm in enumerate(md['trace_name'])}
    log.info("%s mapping ➜ %d keys ; first 15: %s",
             type(ds_like).__name__, len(mapping), list(mapping)[:15])

    def lookup(self, trace_name: str, **kw):  # type: ignore
        if trace_name in mapping:
            idx = mapping[trace_name]
            log.debug("[%s] exact   %-25s -> %4d", type(self).__name__, trace_name, idx)
            return idx
        alt = f"{trace_name}_S.UNKNOWN"
        if alt in mapping:
            idx = mapping[alt]
            log.debug("[%s] suffix  %-25s -> %4d", type(self).__name__, alt, idx)
            return idx
        pref = [k for k in mapping if k.startswith(trace_name + "_")]
        if pref:
            idx = mapping[pref[0]]
            log.debug("[%s] prefix  %-25s -> %4d (via %s)", type(self).__name__, trace_name, idx, pref[0])
            return idx
        log.debug("[%s] introuvable %-25s (aucun préfixe)", type(self).__name__, trace_name)
        raise KeyError(trace_name)

    ds_like.get_idx_from_trace_name = types.MethodType(lookup, ds_like)
    return ds_like

###############################################################################
# Fabrique de dataset simulé + prints complets ↴
###############################################################################

def _simu_data_factory(*args, **kwargs):
    ds = WaveformDataset(DATA_DIR, *args, **kwargs)

    log.info("WaveformDataset loaded ➜ %d traces", len(ds))
    log.info("Metadata columns : %s", list(ds.metadata.columns))
    log.info("Metadata head:\n%s", ds.metadata.head())

    # inspection HDF5 interne
    with h5py.File(DATA_DIR / "waveforms.hdf5", 'r') as f:
        top = list(f.keys())
        log.info("HDF5 top-level groups: %s", top)
        for grp in top:
            keys = list(f[grp].keys())
            log.info("  %s keys (first 10): %s", grp, keys[:10])

    # forcer le split test
    ds.metadata['split'] = 'test'
    _patch_lookup_verbose(ds)

    orig = ds.get_split
    def wrapped(self, name):  # type: ignore
        cd = orig(name)
        log.info("--- get_split('%s') called → CombinedDataset len=%d", name, len(cd))
        if hasattr(cd, 'metadata'):
            log.info("Split metadata columns: %s", list(cd.metadata.columns))
            log.info("Split metadata head:\n%s", cd.metadata.head(5))
        if hasattr(cd, 'datasets'):
            log.info("  subdatasets sizes: %s", [len(sd) for sd in cd.datasets])
        _patch_lookup_verbose(cd)
        return cd

    ds.get_split = types.MethodType(wrapped, ds)  # type: ignore
    return ds

###############################################################################
# Résolveur de noms
###############################################################################

def get_dataset_by_name(name: str):
    if name == 'simulated_data':
        return _simu_data_factory
    if hasattr(sbd, name):
        return getattr(sbd, name)
    raise ValueError(name)

###############################################################################
# save_pick_predictions full debug ↴
###############################################################################

def save_pick_predictions(
    *, model: L.LightningModule, target_path: str, sets: str = 'test',
    batch_size: int = 32, num_workers: int = 0
):
    targets = Path(target_path)
    dataset = _simu_data_factory(component_order='ZNE', dimension_order='NCW', cache='full')

    for split_name in sets.split(','):
        log.info("Processing eval set: %s", split_name)
        split = dataset.get_split(split_name)
        log.info("Preloading waveforms...")
        split.preload_waveforms(pbar=False)

        log.info("Split object: %s", split)
        log.info("Split dir(): %s", dir(split))

        # lecture des targets
        csv = targets / 'task23.csv'
        tt = pd.read_csv(csv)
        tt = tt[tt['trace_split'] == split_name]
        log.info("task23.csv filtered ➜ %d rows", len(tt))
        log.info("task_targets columns: %s", list(tt.columns))

                # instantiate base generator
        gen_base = sbg.SteeredGenerator(split, tt)
        # wrap generator to inject window_borders from CSV
        class GeneratorWithBorders(IterableDataset):  # type: ignore
            def __init__(self, base_gen, targets):
                self.base = base_gen
                # ensure predictable indexing
                self.targets = targets.reset_index(drop=True)
            def __iter__(self):
                for idx, X in enumerate(self.base):
                    row = self.targets.iloc[idx]
                    wb = torch.tensor([row.start_sample, row.end_sample], dtype=torch.long)
                    yield {"X": X, "window_borders": wb}
        gen = GeneratorWithBorders(gen_base, tt)
        log.info("Wrapped generator to include 'window_borders' directly via IterableDataset wrapper")
        loader = DataLoader(gen, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        log.info("DataLoader → %d batches", len(loader))

        # inspect first batch
        try:
            b0 = next(iter(loader))
            log.info("First batch type: %s", type(b0))
            if isinstance(b0, dict):
                log.info("First batch keys: %s", list(b0.keys()))
                log.info("First batch window_borders: %s", b0.get("window_borders", None))
            else:
                log.info("First batch content: %s", b0)
        except Exception as e:
            log.exception("Failed to get first batch: %s", e)

        trainer = L.Trainer(
            accelerator='cpu', logger=False,
            enable_checkpointing=False, limit_predict_batches=1
        )
        try:
            trainer.predict(model, loader)
        except Exception as exc:
            log.exception("Exception during predict: %s", exc)
            raise
