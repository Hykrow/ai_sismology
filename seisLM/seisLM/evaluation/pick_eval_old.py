"""  Module for evaluating phase-picking performance.

https://github.com/seisbench/pick-benchmark
"""
print("✅ pick_eval.py bien exécuté")
import logging
from typing import Optional, Dict
from pathlib import Path
import logging
import numpy as np
from sklearn import metrics
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
import seisbench.data as sbd
import h5py
import seisbench.generate as sbg
from seisLM.utils import project_path
import types          # ← ajoute-le ici
DATA_DIR = Path("/home/noam/seisLM/inference/simulated_data")
from seisbench.data.base import WaveformDataset

        
# utils.py  (ou où tu préfères définir la fonction)
from pathlib import Path
from functools import partial
import seisbench.data as sbd
from seisbench.data import WaveformDataset

# Dossier contenant waveforms.hdf5 + metadata.csv
DATA_DIR = Path("/home/noam/seisLM/inference/simulated_data")

def _simu_data_factory(*args, **kwargs):
    ds = WaveformDataset(DATA_DIR, *args, **kwargs)
    
    if "split" not in ds.metadata or ds.metadata["split"].eq("").all():
        ds.metadata["split"] = "test"

    _patch_lookup_method(ds)          # ← patch du dataset racine
    return ds


import types
import pandas as pd
import seisbench.data as sbd

def _patch_lookup_method(ds_like):
    """
    Patche n'importe quel dataset (WaveformDataset, CombinedDataset…)
    pour qu'il ne cherche les traces qu'à partir du nom, sans dépasser
    les bornes de metadata.
    """
    import types

    # -- Réinitialise l’index du DataFrame *en place* -----------------
    ds_like._metadata.reset_index(drop=True, inplace=True)

    # -- Table trace_name -> position --------------------------------
    name_to_pos = {n: i for i, n in enumerate(ds_like._metadata["trace_name"])}

    # -- Nouvelle méthode de lookup ----------------------------------
    def lookup(self, *a, **kw):
        trace = kw.get("trace_name") or (a[0] if a else None)
        if trace is None:
            raise KeyError("trace_name manquant")

        if trace in name_to_pos:
            return name_to_pos[trace]

        alt = f"{trace}_S.UNKNOWN"
        if alt in name_to_pos:
            return name_to_pos[alt]

        pref = [k for k in name_to_pos if k.startswith(trace + "_")]
        if len(pref) == 1:
            return name_to_pos[pref[0]]

        raise KeyError(f"Trace '{trace}' introuvable")

    # -- Injection ----------------------------------------------------
    ds_like.get_idx_from_trace_name = types.MethodType(lookup, ds_like)
    return ds_like


def get_dataset_by_name(name: str):
    """
    Resolve dataset name to class/factory accepted by save_pick_predictions.
    :param name: Name of dataset as defined in seisbench.data, or "simu_data".
    :return: A class or factory that can be called with () to build a dataset.
    """
    # 1. Data sets “natifs” de SeisBench
    if hasattr(sbd, name):
        return getattr(sbd, name)

    # 2. Jeu simulé maison
    if name == "simu_data":
        # On renvoie la fabrique, **pas l’instance directement** !
        return _simu_data_factory

    # 3. Nom inconnu
    raise ValueError(f"Unknown dataset '{name}'.")


def _identify_instance_dataset_border(task_targets: Dict) -> int:
  """
  Calculates the dataset border between Signal and Noise for instance,
  assuming it is the only place where the bucket number does not increase
  """
  buckets = task_targets["trace_name"].apply(lambda x: int(x.split("$")[0][6:]))

  last_bucket = 0
  for i, bucket in enumerate(buckets):
    if bucket < last_bucket:
      return i
    last_bucket = bucket



def save_pick_predictions(
  model: L.LightningModule,
  target_path: str,
  sets: str,
  save_tag: str,
  batch_size: int = 1024,
  num_workers: int = 4,
  sampling_rate: Optional[int] = None,
  ) -> None:
  targets = Path(target_path)

  sets = sets.split(",")
  model.eval()

  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

  dataset = get_dataset_by_name(targets.name)(
      sampling_rate=100, component_order="ZNE", dimension_order="NCW",
      cache="full"
  )


  #dataset.metadata[]
  pred_root = Path(project_path.EVAL_SAVE_DIR) / f"{save_tag}_{targets.name}"
  weight_path_name = pred_root.name 

  if sampling_rate is not None:
    dataset.sampling_rate = sampling_rate
    pred_root = pred_root.parent / (pred_root.name + "_resampled")
    weight_path_name = weight_path_name + f"_{sampling_rate}"

  for eval_set in sets:
    split = dataset.get_split(eval_set)
    if targets.name == "InstanceCountsCombined":
      logging.warning(
          "Overwriting noise trace_names to allow correct identification"
      )
      # Replace trace names for noise entries
      split._metadata["trace_name"].values[
          -len(split.datasets[-1]) :
      ] = split._metadata["trace_name"][-len(split.datasets[-1]) :].apply(
          lambda x: "noise_" + x
      )
      split._build_trace_name_to_idx_dict()

    logging.warning(f"Starting set {eval_set}")
    split.preload_waveforms(pbar=True)

    for task in ["1", "23"]: #REMPLACER PAR 1, 23

      task_csv = targets / f"task{task}.csv"
      if not task_csv.is_file():
        print("fonctionne pas", flush=True)
        continue

      logging.warning(f"Starting task {task}")

      task_targets = pd.read_csv(task_csv)
      task_targets = task_targets[task_targets["trace_split"] == eval_set]
      # Filtrer les entrées avec une fenêtre trop grande
      print("task target end sample", task_targets["end_sample"])
      print("task target start sample", task_targets["start_sample"])
      #print("duree de la window : ", task_targets["end_sample"] - task_targets["start_sample"])
      #task_targets = task_targets[
        #  (task_targets["end_sample"] - task_targets["start_sample"]) <= 3001
      #]

      if task == "1" and targets.name == "InstanceCountsCombined":
        border = _identify_instance_dataset_border(task_targets)
        task_targets["trace_name"].values[border:] = task_targets["trace_name"][
            border:
        ].apply(lambda x: "noise_" + x)
      if 'sampling_rate' not in task_targets.columns:
          if 'trace_sampling_rate_hz' in task_targets.columns:
              task_targets['sampling_rate'] = task_targets['trace_sampling_rate_hz']
          elif 'trace_dt_s' in task_targets.columns:
              task_targets['sampling_rate'] = 1.0 / task_targets['trace_dt_s']
          else:
              raise KeyError(
                  "Aucune colonne 'sampling_rate', 'trace_sampling_rate_hz' ou 'trace_dt_s' trouvée"
              )
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
     # print("dataset metadata: ", dataset.metadata["trace_name"])
      #print("KEY DATASET :", list(dataset._hdf5_file.keys()))
      #dataset._metadata["trace_name"] = (
      #    dataset._metadata["trace_name_original"].fillna("").astype(str)
      #)
     # task_targets["trace_name"] = task_targets["trace_name_original"]
      generator = sbg.SteeredGenerator(split, task_targets)
      generator.add_augmentations(model.get_eval_augmentations())

      loader = DataLoader(
        generator, batch_size=batch_size, shuffle=False, num_workers=num_workers
      )
      trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,            # Disable the default logger
        enable_checkpointing=False  # Disable automatic checkpointing
      )

      print(">>> DÉBUT PREDICT")
      print("TASKS TARGET  : ", task_targets['trace_name'])

      predictions = trainer.predict(model, loader)
      print(">>> PRÉDICTIONS FAITES")
      print("Nombre d'outputs de prédiction :", len(predictions[0]))
      # Merge batches
      merged_predictions = []

      for i, _ in enumerate(predictions[0]):
        merged_predictions.append(torch.cat([x[i] for x in predictions]))

      merged_predictions = [x.cpu().numpy() for x in merged_predictions]
      #merged_predictions = [
      #  torch.cat([batch[i] for batch in predictions]).cpu().numpy()
      #  for i in range(len(predictions[0]))
      #]
      task_targets["score_detection"] = merged_predictions[0]
      task_targets["score_p_or_s"] = merged_predictions[1]
      task_targets["p_sample_pred"] = (
          merged_predictions[2] + task_targets["start_sample"]
      )
      task_targets["s_sample_pred"] = (
          merged_predictions[3] + task_targets["start_sample"]
      )


      pred_path = (
        Path(project_path.EVAL_SAVE_DIR)
        / f"{save_tag}_{targets.name}"
        / f"{eval_set}_task{task}.csv"
      )
      pred_path.parent.mkdir(exist_ok=True, parents=True)
      # pred_path = f'./{eval_set}_task{task}.csv'
      logging.warning(f"Saving predictions to {pred_path}")
      task_targets.to_csv(pred_path, index=False)


def get_results_event_detection(pred_path):
  pred = pd.read_csv(pred_path)
  pred["trace_type_bin"] = pred["trace_type"] == "earthquake"
  pred["score_detection"] = pred["score_detection"].fillna(0)

  fpr, tpr, _ = metrics.roc_curve(
    pred["trace_type_bin"], pred["score_detection"])
  prec, recall, thr = metrics.precision_recall_curve(
    pred["trace_type_bin"], pred["score_detection"]
  )
  auc = metrics.roc_auc_score(
    pred["trace_type_bin"], pred["score_detection"]
  )


  f1 = 2 * prec * recall / (prec + recall)
  f1_threshold = thr[np.nanargmax(f1)]
  best_f1 = np.max(f1)

  return {
    'auc': auc,
    'fpr': fpr,
    'tpr': tpr,
    'prec': prec,
    'recall': recall,
    'f1': f1,
    'f1_threshold': f1_threshold,
    'best_f1': best_f1
  }

def get_results_phase_identification(pred_path):
  pred = pd.read_csv(pred_path)
  print(pred)
  pred["phase_label_bin"] = pred["phase_label"] == "P"
  pred["score_p_or_s"] = pred["score_p_or_s"].fillna(0)
  fpr, tpr, _ = metrics.roc_curve(
    pred["phase_label_bin"], pred["score_p_or_s"]
  )
  prec, recall, thr = metrics.precision_recall_curve(
    pred["phase_label_bin"], pred["score_p_or_s"]
  )
  f1 = 2 * prec * recall / (prec + recall)
  f1_threshold = thr[np.nanargmax(f1)]
  best_f1 = np.nanmax(f1)

  auc = metrics.roc_auc_score(
    pred["phase_label_bin"], pred["score_p_or_s"]
  )

  return {
    'auc': auc,
    'fpr': fpr,
    'tpr': tpr,
    'prec': prec,
    'recall': recall,
    'f1': f1,
    'f1_threshold': f1_threshold,
    'best_f1': best_f1
  }

def get_results_onset_determination(pred_path):
  pred = pd.read_csv(pred_path)
  results = {}
  for phase in ['P', 'S']:
    pred_phase = pred[pred["phase_label"] == phase]
    pred_col = f"{phase.lower()}_sample_pred"
    diff = (pred_phase[pred_col] - pred_phase["phase_onset"]
            ) / pred_phase["sampling_rate"]
    results[f'{phase}_onset_diff'] = diff
  return results
