import torch
torch.set_float32_matmul_precision('high')
from seisLM.evaluation.pick_eval import save_pick_predictions
from seisLM.model.task_specific.phasepick_models import MultiDimWav2Vec2ForFrameClassificationLit, PhaseNetLit  # Ton modèle entraîné
from pathlib import Path
from seisbench.data import ETHZ
dataset = ETHZ()



# Charger ton modèle entraîné
model = PhaseNetLit.load_from_checkpoint("/home/noam/seisLM/results/models/phasepick_run/ethz_phasenet__train_frac_0.7_time_2025-04-06-19h-01m-45s/checkpoints/epoch=10-step=5258.ckpt")

#model = PhaseNetLit.load_from_checkpoint("/home/noam/seisLM/results/models/phasepick_run/ethz_phasenet__train_frac_1.0_time_2025-04-05-23h-31m-43s/checkpoints/epoch=48-step=34643.ckpt")
model.eval()

# Générer les prédictions
save_pick_predictions(
    model=model,
    target_path="/home/noam/seisLM/data/targets/ETHZ/",
    sets="test",
    save_tag="ethz_inference",
    batch_size=128,
    num_workers=4,
    sampling_rate=100,  # ou None si inchangé
)







"""
model = MultiDimWav2Vec2ForFrameClassificationLit.load_from_checkpoint("./pretrained/model.ckpt")
model.eval()

# Générer les prédictions
save_pick_predictions(
    model=model,
    target_path="ETHZ",
    sets="test",
    save_tag="ethz_inference",
    batch_size=128,
    num_workers=4,
    sampling_rate=100,  # ou None si inchangé
)
"""