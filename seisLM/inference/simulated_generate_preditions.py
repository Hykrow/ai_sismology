#!/usr/bin/env python3
"""simulated_generate_predictions.py — v2.2
───────────────────────────────────────────────────────────────────
Version précédente : toujours un *IndexError* car l’index calculé pouvait
être décalé après que SeisBench reconstruise son propre dictionnaire interne
`_trace_name_to_idx`.

### Correctif v2.2 — lookup **toujours synchronisé**
* Dans `_patch_lookup` on **n’utilise plus** de tableau pré‑calculé.
* On interroge directement `self._trace_name_to_idx`, et si ce dict n’existe
  pas ou qu’il est vide, on appelle `self._build_trace_name_to_idx_dict()`.
* Ainsi l’indice renvoyé est garanti cohérent avec l’état courant de
  `metadata`, même après des opérations internes (concat, preload, etc.).

Le reste est inchangé ; lance simplement :
```bash
python simulated_generate_predictions.py
```"""
from __future__ import annotations

import argparse
import logging
import types
from pathlib import Path

import pandas as pd
import torch
from seisLM.evaluation.pick_eval_debug import save_pick_predictions   # au lieu de pick_eval

import seisbench.data as sbd
from seisbench.data import WaveformDataset
from seisLM.model.task_specific.phasepick_models import MultiDimWav2Vec2ForFrameClassificationLit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("simu")

###############################################################################
# Patch helpers
###############################################################################

def _ensure_trace_dict(ds_like):
    """(Re)construire le dict interne trace→idx si nécessaire."""
    if not getattr(ds_like, "_trace_name_to_idx", None):
        if hasattr(ds_like, "_build_trace_name_to_idx_dict"):
            ds_like._build_trace_name_to_idx_dict()


def _lookup_factory(ds_like):
    """Crée une méthode lookup synchronisée ET verbeuse pour debug."""
    def lookup(self, *a, **kw):  # type: ignore[no-self-arg]
        trace = kw.get("trace_name") or (a[0] if a else None)
        if trace is None:
            raise KeyError("trace_name manquant")

        _ensure_trace_dict(self)
        mapping = self._trace_name_to_idx  # type: ignore[attr-defined]

        # helper pour print
        def _dbg(msg: str, idx: int | None = None):
            #log.debug("[%s] %s | len(meta)=%d | idx=%s", self.__class__.__name__, msg, len(self._metadata), idx)
            pass
        # exact
        if trace in mapping:
            _dbg("exact", mapping[trace])
            return mapping[trace]
        # alt suffix
        alt = f"{trace}_S.UNKNOWN"
        if alt in mapping:
            _dbg("suffix", mapping[alt])
            return mapping[alt]
        # unique prefix
        pref = [k for k in mapping if k.startswith(trace + "_")]
        if len(pref) == 1:
            _dbg("prefix", mapping[pref[0]])
            return mapping[pref[0]]
        _dbg("introuvable")
        raise KeyError(f"Trace '{trace}' introuvable")

    return types.MethodType(lookup, ds_like)


def _patch_lookup(ds_like):
    """Applique la nouvelle méthode lookup et réinitialise l’index."""
    md: pd.DataFrame = ds_like._metadata  # type: ignore[attr-defined]
    md.reset_index(drop=True, inplace=True)
    ds_like.get_idx_from_trace_name = _lookup_factory(ds_like)
    return ds_like


def _patch_recursive(ds_like):
    _patch_lookup(ds_like)
    if hasattr(ds_like, "datasets"):
        for sub in ds_like.datasets:
            _patch_lookup(sub)
    return ds_like

###############################################################################
# Simulated dataset factory
###############################################################################

def _simu_factory(data_dir: Path, *args, **kwargs):
    kwargs.setdefault("cache", "full")
    base = WaveformDataset(data_dir, *args, **kwargs)

    if "split" not in base.metadata.columns or base.metadata["split"].eq("").all():
        base.metadata["split"] = "test"

    _patch_lookup(base)

    # Wrap get_split so CombinedDataset est patché à chaud
    orig_get_split = base.get_split

    def wrapped_get_split(self, split_name):  # type: ignore[no-self-arg]
        cd = orig_get_split(split_name)
        _patch_recursive(cd)
        return cd

    base.get_split = types.MethodType(wrapped_get_split, base)  # type: ignore[assignment]
    return base

###############################################################################
# Monkey‑patch pick_eval.get_dataset_by_name
###############################################################################


###############################################################################
# Main
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/noam/seisLM/results/models/phasepick_run/ethz_seisLM__train_frac_0.7_time_2025-05-18-16h-17m-48s/checkpoints/model.ckpt")
    ap.add_argument("--data-dir", default="/home/noam/seisLM/inference/simulated_data")
    ap.add_argument("--save-tag", default="sim_inference")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    torch.set_float32_matmul_precision("high")
    print("args ckpt", args.ckpt)
    model = MultiDimWav2Vec2ForFrameClassificationLit.load_from_checkpoint(args.ckpt)
    model.eval()


    # Répertoire où se trouve ce fichier .py
    BASE_DIR = Path(__file__).resolve().parent

    # Le dossier simulated_data, toujours à côté de votre script
    DATA_DIR = BASE_DIR / "simulated_data"

    save_pick_predictions(
        model=model,
        target_path=DATA_DIR,
        batch_size=64
    )


if __name__ == "__main__":
    main()
