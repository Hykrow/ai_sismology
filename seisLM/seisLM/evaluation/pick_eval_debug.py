# -*- coding: utf-8 -*-
"""pick_eval_mod.py — v0.14
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
from torch.utils.data import DataLoader, Dataset
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
    #log.info("%s mapping ➜ %d keys ; first 15: %s",
     #        type(ds_like).__name__, len(mapping), list(mapping)[:15])

    def lookup(self, trace_name: str, **kw):
        if trace_name in mapping:
            idx = mapping[trace_name]
            #log.debug("[%s] exact   %-25s -> %4d", type(self).__name__, trace_name, idx)
            return idx
        alt = f"{trace_name}_S.UNKNOWN"
        if alt in mapping:
            idx = mapping[alt]
            #log.debug("[%s] suffix  %-25s -> %4d", type(self).__name__, alt, idx)
            return idx
        pref = [k for k in mapping if k.startswith(trace_name + "_")]
        if pref:
            idx = mapping[pref[0]]
            #log.debug("[%s] prefix  %-25s -> %4d (via %s)", type(self).__name__, trace_name, idx, pref[0])
            return idx
        #log.debug("[%s] introuvable %-25s (aucun préfixe)", type(self).__name__, trace_name)
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

    with h5py.File(DATA_DIR / "waveforms.hdf5", 'r') as f:
        top = list(f.keys())
        log.info("HDF5 top-level groups: %s", top)
        for grp in top:
            keys = list(f[grp].keys())
            log.info("  %s keys (first 10): %s", grp, keys[:10])

    ds.metadata['split'] = 'test'
    _patch_lookup_verbose(ds)

    orig = ds.get_split
    def wrapped(self, name):
        cd = orig(name)
        log.info("--- get_split('%s') called → CombinedDataset len=%d", name, len(cd))
        if hasattr(cd, 'metadata'):
            log.info("Split metadata columns: %s", list(cd.metadata.columns))
            log.info("Split metadata head:\n%s", cd.metadata.head(5))
        if hasattr(cd, 'datasets'):
            log.info("  subdatasets sizes: %s", [len(sd) for sd in cd.datasets])
        _patch_lookup_verbose(cd)
        return cd

    ds.get_split = types.MethodType(wrapped, ds)
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
    batch_size: int = 32, num_workers: int = 4
):
    targets = Path(target_path)
    dataset = _simu_data_factory(component_order='ZNE', dimension_order='NCW', cache='full')

    for split_name in sets.split(','):
        log.info("Processing eval set: %s", split_name)
        split = dataset.get_split(split_name)
        log.info("Preloading waveforms...")
        split.preload_waveforms(pbar=False)

        csv = targets / 'task23.csv'
        tt = pd.read_csv(csv)
        tt = tt[tt['trace_split'] == split_name].reset_index(drop=True)
        log.info("task23.csv filtered ➜ %d rows", len(tt))

        gen_base = sbg.SteeredGenerator(split, tt)

        class GeneratorWithBorders(Dataset):
            def __init__(self, base_gen, targets):
                self.base = base_gen
                self.targets = targets

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                sample = self.base[idx]
                if isinstance(sample, dict) and 'X' in sample:
                    x = sample['X']
                else:
                    x = sample
                if not torch.is_tensor(x):
                    x = torch.as_tensor(x, dtype=next(model.parameters()).dtype)
                else:
                    x = x.to(dtype=next(model.parameters()).dtype)
                row = self.targets.iloc[idx]
                start, end = int(row.start_sample), int(row.end_sample)
                c, L_ = x.shape
                seg_len = end - start
                x_seg = x.new_zeros((c, seg_len))
                src_start = max(0, start)
                src_end = min(L_, end)
                dst_start = max(0, -start)
                dst_end = dst_start + (src_end - src_start)
                x_seg[:, dst_start:dst_end] = x[:, src_start:src_end]
                wb = torch.tensor([0, seg_len], dtype=torch.long)
                return {"X": x_seg, "window_borders": wb}

        gen = GeneratorWithBorders(gen_base, tt)
        loader = DataLoader(gen, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        log.info("DataLoader → %d batches", len(loader))

        try:
            b0 = next(iter(loader))
            log.info("First batch keys: %s", list(b0.keys()) if isinstance(b0, dict) else b0)
        except Exception as e:
            log.exception("Failed to get first batch: %s", e)

        trainer = L.Trainer(
            accelerator='gpu', logger=False,
            enable_checkpointing=False
        )
        try:
            predictions = trainer.predict(model, loader)
            print("predictions", predictions)
            task_targets = tt
            # Filtrer les entrées avec une fenêtre trop grande
            print("task target end sample", task_targets["end_sample"])
            print("task target start sample", task_targets["start_sample"])
            #print("duree de la window : ", task_targets["end_sample"] - task_targets["start_sample"])
            #task_targets = task_targets[
                #  (task_targets["end_sample"] - task_targets["start_sample"]) <= 3001
            #]


            if 'sampling_rate' not in task_targets.columns:
                if 'trace_sampling_rate_hz' in task_targets.columns:
                    task_targets['sampling_rate'] = task_targets['trace_sampling_rate_hz']
                elif 'trace_dt_s' in task_targets.columns:
                    task_targets['sampling_rate'] = 1.0 / task_targets['trace_dt_s']
                else:
                    raise KeyError(
                        "Aucune colonne 'sampling_rate', 'trace_sampling_rate_hz' ou 'trace_dt_s' trouvée"
                    )
            sampling_rate = 100
            if sampling_rate is not None:
                for key in ["start_sample", "end_sample", "phase_onset"]:
                    if key not in task_targets.columns:
                        continue
                task_targets[key] = (
                    task_targets[key]
                    * sampling_rate
                    / task_targets["sampling_rate"]
                )
                task_targets[sampling_rate] = sampling_rate
            merged_predictions = []

            for i, _ in enumerate(predictions[0]):
                merged_predictions.append(torch.cat([x[i] for x in predictions]))

            merged_predictions = [x.cpu().numpy() for x in merged_predictions]
            #merged_predictions = [
            #  torch.cat([batch[i] for batch in predictions]).cpu().numpy()
            #  for i in range(len(predictions[0]))
            #]
            #print("score detection", task_targets["score_detection"])
            print("merged predictions", merged_predictions[0])
            task_targets["score_detection"] = merged_predictions[0]
            task_targets["score_p_or_s"] = merged_predictions[1]
            task_targets["p_sample_pred"] = (
                merged_predictions[2] + task_targets["start_sample"]
            )
            task_targets["s_sample_pred"] = (
                merged_predictions[3] + task_targets["start_sample"]
            )


            pred_path = (
                Path("/home/noam/seisLM/predictions/preds.csv")
            )
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            # pred_path = f'./{eval_set}_task{task}.csv'
            logging.warning(f"Saving predictions to {pred_path}")
            task_targets.to_csv(pred_path, index=False)
        except Exception as exc:
            log.exception("Exception during predict: %s", exc)
            raise
