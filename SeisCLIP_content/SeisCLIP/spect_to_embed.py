import os
import numpy as np
# Forcer torch à utiliser ton répertoire perso pour les checkpoints
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')


from Pretrain.model_seismic_clip_two_branch import AUDIO_CLIP
import torch, h5py, torch.nn.functional as F
import torchaudio
from pathlib import Path

def compute_spectrogram(trace, n_fft=128, hop_length=2, n_mels=50):
    waveform = torch.tensor(trace).unsqueeze(0)  # (1, samples)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=100,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    return spec_db.squeeze(0).numpy() 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AUDIO_CLIP(
    device_name = device,
    embed_dim = 384,
    text_input = 8,
    text_width = 512,
    text_layers = 2,
    spec_fdim = 50,          # 50 bandes de fréquence (n_mels)
    spec_tdim = 600,         # 600 frames temporelles
    spec_model_size = 'small224',
    imagenet_pretrain = True,
).to(device)

print("spec.conv1.weight.shape in model instance :", model.spec.conv1.weight.shape)

# Charger les poids du checkpoint

state_dict = torch.load('checkpoints/SeisCLIP_STEAD_50x600.ckpt', map_location='cpu')

#print("Modèle AUDIO_CLIP modules :", dict(model.named_modules()).keys())


# Nettoyage éventuel des préfixes (à adapter si besoin)
#new_state_dict = {}
#for k, v in state_dict.items():
    # Ici tu peux gérer les préfixes selon la structure de ton AUDIO_CLIP
    # Exemple : enlever un "spec." devant chaque clé si nécessaire
#    new_state_dict[k] = v  # Pas de modif pour l’instant

# Charger le state_dict dans le modèle

model.load_state_dict(state_dict)

model.eval()                 # AUDIO_CLIP déjà instancié & chargé

# --- charger la première trace ---

DATA_DIR = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data")
H5_FILE  = DATA_DIR / "my_velocity_data.hdf5"
with h5py.File(H5_FILE, "r") as f:
    cube = f["sample100000_x16_y16"][:]          # shape (3, 800)

# --- appliquer ta fonction canal par canal + interpolation 600 frames ---
spec_channels = []
for comp in cube:                                # comp = (800,)
    spec = compute_spectrogram(comp)             # (50, ~337)   ← ta fonction
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)      # (1,1,50,T)
    spec = F.interpolate(spec, size=(50, 600),
                         mode="bilinear", align_corners=False)
    spec_channels.append(spec.squeeze())         # remet en (50,600)

input_tensor = torch.stack(spec_channels).unsqueeze(0).to(device)  # (1,3,50,600)
print("shape entrée :", input_tensor.shape)


with torch.no_grad():
    emb_audio = model.encode_audio(input_tensor)          # (1, 384)
    emb_audio = emb_audio / emb_audio.norm(dim=-1, keepdim=True)

print("Embedding obtenu – norme :", emb_audio.norm().item())  # ≈ 1.0 si tout va bien


def trace_to_tensor(cube):
    """
    cube : np.ndarray (3, 800)  Z, N, E
    ↳ retourne torch.Tensor (3, 50, 600)
    """
    specs = []
    for comp in cube:                               # boucle sur Z, N, E
        spec = compute_spectrogram(comp)            # (50, ~337)  ← TA fonction
        spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)     # (1,1,50,T)
        spec = F.interpolate(spec, size=(50, 600),
                             mode="bilinear", align_corners=False)
        specs.append(spec.squeeze())                # (50,600)
    return torch.stack(specs)                       # (3, 50, 600)


embeddings   = []          # pour stocker (N, 384)
trace_names  = []          # pour garder l’ordre
BATCH        = 128         # adapter à la mémoire GPU

with h5py.File(H5_FILE, "r") as f:
    names = list(f.keys())
    for i in range(0, len(names), BATCH):
        batch_tensors = []
        for name in names[i:i+BATCH]:
            cube  = f[name][:]                    # (3, 800)
            spec  = trace_to_tensor(cube)         # (3, 50, 600)
            batch_tensors.append(spec)
            trace_names.append(name)

        batch = torch.stack(batch_tensors).to(device)       # (B,3,50,600)

        with torch.no_grad():
            emb = model.encode_audio(batch)                 # (B, 384)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu())

# concatène tous les lots
embeddings = torch.cat(embeddings)   # shape (N_traces, 384)
print("embeddings finaux :", embeddings.shape)

# convertir en numpy pour affichage plus compact
emb_np = embeddings.cpu().numpy()

# 1)  sauvegarde NumPy
'''np.save("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/velocity_emb_with SeisCLIP_384.npy",
        emb_np)'''
np.save("velocity_emb_with SeisCLIP_384.npy",
        emb_np)
with open("trace_names.txt", "w") as f:
    f.writelines(n + "\n" for n in trace_names)


# afficher les 3 premiers vecteurs, tronqués aux 10 premières dimensions
np.set_printoptions(precision=4, suppress=True)
print("\n--- aperçu embeddings (3 traces × 10 dims) ---")
print(emb_np[:3, :10])
