import matplotlib.pyplot as plt
import h5py

file_path = "velocity/velocity100000-100099/sample100000.h5"

with h5py.File(file_path, "r") as f:
    uZ = f["uZ"][16, 16, :]
    uN = f["uN"][16, 16, :]
    uE = f["uE"][16, 16, :]

plt.figure(figsize=(14, 6))
plt.plot(uZ, label="uZ (Vertical)")
plt.plot(uN, label="uN (North-South)")
plt.plot(uE, label="uE (East-West)")
plt.title("Composantes - Capteur (16,16) - sample100000")
plt.xlabel("Temps (échantillons)")
plt.ylabel("Vitesse (m/s)")
plt.legend()
plt.grid()
plt.show()
