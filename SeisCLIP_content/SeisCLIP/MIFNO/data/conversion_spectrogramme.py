import numpy as np
import torch
import h5py
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm


def compute_spectrogram(trace, n_fft=128, hop_length=2, n_mels=50):
    waveform = torch.tensor(trace).unsqueeze(0)  # (1, samples)
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=100,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    return spec_db.squeeze(0).numpy()  # (n_mels, time)

# Exemple : charger une trace velocity
with h5py.File("my_velocity_data.hdf5", "r") as f:
    trace_data = f["sample100000_x16_y16"][:]  # (3, 800)

# Calcul du spectrogramme pour Z, N, E
specs = []
for i in range(3):
    spec = compute_spectrogram(trace_data[i], n_fft=128, hop_length=2, n_mels=50)  # (50, t)
    specs.append(spec[:, :600])  # tronquer à 600 frames temporelles

input_tensor = torch.tensor(np.stack(specs)).unsqueeze(0)  # (1, 3, 50, 600)

# Exemple : spec = (n_mels, 401)
spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, n_mels, 401)

# Interpolation vers 600 frames temporelles
spec_resized = F.interpolate(spec, size=(spec.shape[2], 600), mode="bilinear", align_corners=False)

# Remove batch & channel dims
spec_resized = spec_resized.squeeze(0).squeeze(0).numpy()  # shape: (n_mels, 600)


print("Nouvelle shape :", spec_resized.shape)

# Affichage des infos :
print("Shape du spectrogramme :", spec_resized.shape)
print("Valeurs min/max :", spec_resized.min(), spec_resized.max())

# Visualisation en image
import matplotlib.pyplot as plt
plt.imshow(spec_resized, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="dB")
plt.xlabel("Temps (frames)")
plt.ylabel("Fréquence (Mel bins)")
plt.title("Spectrogramme Mel d'une trace")
plt.show()
