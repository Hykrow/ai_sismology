#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune de la branche texte (Info_embedding) de SeisCLIP
sur les spectrogrammes HEMEW-S-3D.
"""
import h5py
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------
# 1. chemins & hyper-paramètres
# ------------------------------------------------------------------
PRETRAINED   = Path("checkpoints/SeisCLIP_STEAD_50x600.ckpt")
SNAPSHOT_BEFORE = Path("checkpoints/pre_finetune.ckpt")
SNAPSHOT_AFTER  = Path("checkpoints/finetuned_ckpt.pt")


H5_FOLDER   = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data/velocity/velocity100000-100099")          # contient les .h5
META_CSV  = Path("/usr/users/seismofoundation2/sarcos_fra/MIFNO/data/my_velocity_metadata.csv")
TRACE_LST = Path("/usr/users/seismofoundation2/sarcos_fra/SeisCLIP/trace_names.txt")

all_names = [l.strip() for l in open(TRACE_LST)]

# -----------------------------  SPLIT  ----------------------------
np.random.seed(42)              # reproductible
np.random.shuffle(all_names)

N       = len(all_names)
n_train = int(0.8 * N)          # 80 % train
n_val   = int(0.1 * N)          # 10 % val
# reste → test
train_names = all_names[:n_train]
val_names   = all_names[n_train:n_train + n_val]
test_names  = all_names[n_train + n_val:]

print(f"Train : {len(train_names)}  Val : {len(val_names)}  Test : {len(test_names)}")


BATCH   = 64
EPOCHS  = 3
LR      = 1e-3
NEW_TEXT_DIM = 8          # ou 10/12 si tu ajoutes les features matériaux

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 2. fonctions utilitaires  (spectrogramme, vecteur texte)
# ------------------------------------------------------------------
def lowpass_filter(x, cutoff=5.0, fs=100.0, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs; b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, x)

def compute_spectrogram(trace):
    import torchaudio, numpy as np, torch

    # 0) passe-bas + copie pour enlever les strides négatifs
    filt = lowpass_filter(trace).copy()           # ← copie contiguë

    # 1) normalisation z-score
    wav = torch.tensor(filt, dtype=torch.float32) # (T,)
    wav = (wav - wav.mean()) / (wav.std() + 1e-3)
    wav = wav.unsqueeze(0)                        # (1,T)

    # 2) log-Mel 50 × 120, bande 0-5 Hz
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=100,
        n_fft=100, hop_length=50,
        n_mels=50, f_min=0., f_max=5.0
    )(wav)
    mel = torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0)  # (50, time)

    return mel[:, :120].clone().contiguous()         # tronque à 120 trames
    
def build_text_vector(row):
    # ← adapte si tu ajoutes des features matériaux
    return np.array([
        row.source_x/9600, row.source_y/9600, abs(row.source_z)/9000,
        (row.strike%360)/360, row.dip/90,
        ((row.rake+180)%360)/360,
        0., 0.                                # 2 places libres
    ], dtype=np.float32)

# ------------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------------
class FineTuneDataset(Dataset):
    def __init__(self,names):
        self.names = names
        self.meta  = pd.read_csv(META_CSV).set_index("trace_name")
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        sample_id, x_tag, y_tag = name.split('_')       # 'sample100045', 'x18', 'y05'
        x_idx = int(x_tag[1:]);  y_idx = int(y_tag[1:]) # 18, 5

        h5_path = H5_FOLDER / f"{sample_id}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)

        # ---- lecture 3 composantes sur le capteur (x_idx,y_idx) ----
        with h5py.File(h5_path, 'r') as f:
            cube = np.stack([
                f['uZ'][y_idx, x_idx],   # (800,)
                f['uN'][y_idx, x_idx],
                f['uE'][y_idx, x_idx]
            ])                           # (3,800)

        spec = torch.stack([torch.tensor(compute_spectrogram(cube[i]))
                            for i in range(3)])          # (3,50,120)

        text = torch.tensor(build_text_vector(self.meta.loc[name]),
                            dtype=torch.float32)
                            
        return text, spec                      # (8,) , (3,50,120)


# ------------------------------------------------------------------
# 1.2  DataLoaders
# ------------------------------------------------------------------
train_ds = FineTuneDataset(train_names)   # tu passes train_names
val_ds   = FineTuneDataset(val_names)

train_loader = DataLoader(train_ds, batch_size=BATCH,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH,
                          shuffle=False, num_workers=4, pin_memory=True)


# ------------------------------------------------------------------
# 4. Modèle
# ------------------------------------------------------------------
from Pretrain.model_seismic_clip_two_branch import AUDIO_CLIP

model = AUDIO_CLIP(
    device_name = device, embed_dim = 384,
    text_input  = NEW_TEXT_DIM, text_width = 512, text_layers = 2,
    spec_fdim   = 50, spec_tdim = 120, spec_tstr=10, spec_fstr=10,
    spec_model_size='small224', imagenet_pretrain=True,
    load_pretrain_patch=120).to(device)

# ──────────  chargement du checkpoint en ignorant pos_embed  ──────────
state = torch.load(PRETRAINED, map_location='cpu')

# 1) Supprime la clé pos_embed (et FCN_input si NEW_TEXT_DIM ≠ 8)
state = {k: v for k, v in state.items()
         if k != 'spec.v.pos_embed'
         and not k.startswith('info.FCN_input')}

# 2) load_state_dict avec strict=False pour accepter les manques
missing, unexpected = model.load_state_dict(state, strict=False)
print("Clés ignorées :", missing)

# 3) Interpole la nouvelle positional-embedding (46+2 tokens)
with torch.no_grad():
    n_patches = 44        # 50×120 spectro, stride 10×10  → 5×?* ?
    n_tokens  = n_patches + 2            # + cls + dist
    embed_dim = model.spec.v.pos_embed.shape[-1]

    # remplace l’ancien tensor par un nouveau tiré ~N(0,1/√D)
    new_pe = torch.randn(1, n_tokens, embed_dim) * embed_dim**-0.5
    model.spec.v.pos_embed = torch.nn.Parameter(new_pe)

print("pos_embed réinitialisé :", model.spec.v.pos_embed.shape)

# snapshot avant FT
torch.save(model.state_dict(), SNAPSHOT_BEFORE)

# freeze AST
for p in model.spec.parameters(): p.requires_grad = False
# unfreeze texte
for p in model.info.parameters(): p.requires_grad = True

optimizer = torch.optim.Adam(model.info.parameters(), lr=LR)

# ------------------------------------------------------------------
# 5. Boucle d’entraînement
# ------------------------------------------------------------------
model.train()
for epoch in range(1, EPOCHS+1):
    running = 0.0
    for text, spec in tqdm(train_loader, desc=f"E{epoch}/{EPOCHS}"):
        text, spec = text.to(device), spec.to(device)

        audio_emb = model.encode_audio(spec) / model.encode_audio(spec).norm(dim=-1, keepdim=True)
        text_emb  = model.encode_text(text)  / model.encode_text(text).norm(dim=-1, keepdim=True)

        logits = audio_emb @ text_emb.T
        labels = torch.arange(len(logits), device=device)
        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running += loss.item() * len(text)

    train_loss = running / len(train_loader.dataset)

    # -------- phase VALIDATION ----------
    model.eval(); val_running = 0.0
    with torch.no_grad():
        for text, spec in val_loader:
            text, spec = text.to(device), spec.to(device)
            audio_emb = model.encode_audio(spec)
            audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
            text_emb  = model.encode_text(text)
            text_emb  = text_emb  / text_emb.norm(dim=-1, keepdim=True)
            logits = audio_emb @ text_emb.T
            labels = torch.arange(len(logits), device=device)
            val_loss = (F.cross_entropy(logits, labels) +
                        F.cross_entropy(logits.T, labels)) / 2
            val_running += val_loss.item() * len(text)

    val_loss = val_running / len(val_loader.dataset)
    model.train()                       # repasse en mode train

    print(f"E{epoch}  train={train_loss:.4f}  val={val_loss:.4f}")

    # Early-stopping : on garde le meilleur modèle val
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "finetuned_best.pt")

# ------------------------------------------------------------------
# 6. Sauvegarde checkpoint fine-tuned
# ------------------------------------------------------------------
torch.save(model.state_dict(), SNAPSHOT_AFTER)
print("✓ Fine-tuning terminé et sauvegardé :", SNAPSHOT_AFTER)

