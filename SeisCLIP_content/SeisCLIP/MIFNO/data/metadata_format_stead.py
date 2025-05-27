import pandas as pd

source_df = pd.read_csv("metadata/source_properties.csv")

metadata_rows = []

for sample_id in range(100000, 100100):
    row = source_df[source_df["index"] == sample_id].iloc[0]
    
    for x in range(32):
        for y in range(32):
            trace_name = f"sample{sample_id}_x{x:02d}_y{y:02d}"
            metadata_rows.append({
                "trace_name": trace_name,
                "source_x": row["x_s (m)"],
                "source_y": row["y_s (m)"],
                "source_z": row["z_s (m)"],
                "strike": row["strike (°)"],
                "dip": row["dip (°)"],
                "rake": row["rake (°)"]
            })

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv("my_velocity_metadata.csv", index=False)

