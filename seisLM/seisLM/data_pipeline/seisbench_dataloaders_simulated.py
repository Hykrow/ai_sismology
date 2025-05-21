"""Dataloaders for SeisBench datasets."""

import logging
from typing import Any, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from seisbench.data import MultiWaveformDataset
from seisbench.data.base import BenchmarkDataset
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import types
from seisbench.data import WaveformDataset
import h5py
from typing import Dict

data_aliases = {
  "ethz": "ETHZ",
  "geofon": "GEOFON",
  "stead": "STEAD",
  "neic": "NEIC",
  "instance": "InstanceCountsCombined",
  "iquique": "Iquique",
  "lendb": "LenDB",
  "scedc": "SCEDC",
  "simulated_data" : "simulated_data"
}
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
    print("dslike metadata : ", md)
    md.reset_index(drop=True, inplace=True)
    if 'trace_name' not in md.columns and 'name' in md.columns:
        log.warning("Colonne 'trace_name' absente ➜ renomme 'name'")
        md.rename(columns={'name':'trace_name'}, inplace=True)

    mapping: Dict[str,int] = {nm: i for i, nm in enumerate(md['trace_name'])}
    log.info("%s mapping ➜ %d keys ; first 15: %s",
             type(ds_like).__name__, len(mapping), list(mapping)[:15])

    def lookup(self, trace_name: str, **kw):
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

    with h5py.File(DATA_DIR / "waveforms.hdf5", 'r') as f:
        top = list(f.keys())
        log.info("HDF5 top-level groups: %s", top)
        for grp in top:
            keys = list(f[grp].keys())
            log.info("  %s keys (first 10): %s", grp, keys[:10])

    #ds.metadata['split'] = 'test'
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

def apply_training_fraction(
  training_fraction: float,
  train_data: BenchmarkDataset,
) -> None:
  """
  Reduces the size of train_data to train_fraction by inplace filtering.
  Filter blockwise for efficient memory savings.

  Args:
    training_fraction: Training fraction between 0 and 1.
    train_data: Training dataset

  Returns:
    None
  """

  if not 0.0 < training_fraction <= 1.0:
    raise ValueError("Training fraction needs to be between 0 and 1.")

  if training_fraction < 1:
    blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
    unique_blocks = blocks.unique()
    np.random.shuffle(unique_blocks)
    target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
    target_blocks = set(target_blocks)
    mask = blocks.isin(target_blocks)
    train_data.filter(mask, inplace=True)


def prepare_seisbench_dataloaders(
  *,
  model: L.LightningModule,
  data_names: List,
  batch_size: int,
  num_workers: int,
  training_fraction: float = 1.0,
  sampling_rate: int = 100,
  component_order: str = "ZNE",
  dimension_order: str = "NCW",
  collator: Optional[Any] = None,
  cache: Optional[str] = None,
  prefetch_factor: int = 2,
  return_datasets: bool = False,
) -> Union[
  Tuple[DataLoader, DataLoader], Tuple[BenchmarkDataset, BenchmarkDataset]
]:
  """
  Returns the training and validation data loaders
  """
  if isinstance(data_names, str):
    data_names = [data_names]

  multi_waveform_datasets = []
  for data_name in data_names:
    dataset = get_dataset_by_name(data_name)(
      sampling_rate=sampling_rate,
      component_order=component_order,
      dimension_order=dimension_order,
      cache=cache,
    )

    if "split" not in dataset.metadata.columns:
      logging.warning("No split defined, adding auxiliary split.")
      split = np.array(["train"] * len(dataset))
      split[int(0.6 * len(dataset)) : int(0.7 * len(dataset))] = "dev"
      split[int(0.7 * len(dataset)) :] = "test"

      dataset._metadata["split"] = split  # pylint: disable=protected-access

    multi_waveform_datasets.append(dataset)

  if len(multi_waveform_datasets) == 1:
    dataset = multi_waveform_datasets[0]
  else:
    # Concatenate multiple datasets
    dataset = MultiWaveformDataset(multi_waveform_datasets)

  train_data, dev_data = dataset.train(), dataset.dev()
  apply_training_fraction(training_fraction, train_data)

  if cache:
    train_data.preload_waveforms(pbar=True)
    dev_data.preload_waveforms(pbar=True)

  train_generator = sbg.GenericGenerator(train_data)
  dev_generator = sbg.GenericGenerator(dev_data)

  train_generator.add_augmentations(model.get_train_augmentations())
  dev_generator.add_augmentations(model.get_val_augmentations())
  """" DANS LE CAS OU LE GENERATOR NE MARCHE PAS AVEC LE CUSTOM DATASET
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

  gen = GeneratorWithBorders(train_generator, tt)
  """
  train_loader = DataLoader(
    train_generator,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=worker_seeding,
    drop_last=True,  # Avoid crashes from batch norm layers for batch size 1
    pin_memory=True,
    collate_fn=collator,
    prefetch_factor=prefetch_factor,
  )
  dev_loader = DataLoader(
    dev_generator,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=worker_seeding,
    pin_memory=True,
    collate_fn=collator,
    prefetch_factor=prefetch_factor,
  )

  if return_datasets:
    return train_data, dev_data
  else:
    return train_loader, dev_loader
