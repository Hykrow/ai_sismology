import h5py
import numpy as np
import os

output_hdf5 = h5py.File("my_velocity_data.hdf5", "w")

for sample_id in range(100000, 100100):  # fichiers 100000 à 100099
    file_path = f"velocity/velocity100000-100099/sample{sample_id}.h5"
    with h5py.File(file_path, "r") as f:
        uZ = f["uZ"][:]  # (32,32,800)
        uN = f["uN"][:]
        uE = f["uE"][:]
    
    # Crée une trace pour chaque capteur (x, y)
    for x in range(32):
        for y in range(32):
            trace = np.stack([uZ[x, y, :], uN[x, y, :], uE[x, y, :]])  # (3, 800)
            trace_name = f"sample{sample_id}_x{x:02d}_y{y:02d}"
            output_hdf5.create_dataset(trace_name, data=trace)

output_hdf5.close()

