import os
import numpy as np
import pandas as pd
from obspy import read
from tqdm import tqdm

base_dir = "waveforms_data/waveforms"
info_csv = "data/GNSS_M3S_181/info_data.csv"
output_dir = "data/GNSS_M3S_181"
os.makedirs(output_dir, exist_ok=True)

info = pd.read_csv(info_csv)
info['EQ_ID'] = info['EQ_ID'].astype(str).str.zfill(6)

xdata = []
ydata = []
event_ids = []

subductions = [d for d in sorted(os.listdir(base_dir)) if d.startswith("subduction.")]

for event_folder in tqdm(subductions, desc="Traitement des événements"):
    event_id = event_folder.split(".")[-1]
    mag_row = info[info['EQ_ID'] == event_id]
    if mag_row.empty:
        continue
    mag = mag_row['Mag'].values[0]
    event_path = os.path.join(base_dir, event_folder)
    sac_files = [f for f in os.listdir(event_path) if f.endswith(".sac")]
    stations = sorted(set(f.split(".")[0] for f in sac_files))
    if len(stations) < 3:
        continue  # On ne garde que les événements avec au moins 3 stations
    selected_stations = stations[:3]
    station_traces = []
    for station in selected_stations:
        comp_data = []
        for comp, suffix in zip(["N", "E", "Z"], [".LYN.sac", ".LYE.sac", ".LYZ.sac"]):
            sac_path = os.path.join(event_path, station + suffix)
            if os.path.exists(sac_path):
                try:
                    st = read(sac_path)
                    tr = st[0]
                    data = tr.data[:181]
                    if len(data) < 181:
                        data = np.pad(data, (0, 181 - len(data)), 'constant')
                except Exception as e:
                    print(f"Erreur lecture {sac_path}: {e}")
                    data = np.zeros(181)
            else:
                data = np.zeros(181)
            comp_data.append(data)
        station_traces.append(comp_data)  # [3, 181]
    xdata.append(np.stack(station_traces))  # [3, 3, 181]
    ydata.append(mag)
    event_ids.append(event_id)

xdata = np.array(xdata)  # [n_events, 3, 3, 181]
# Transpose pour obtenir (n_events, 3, 181, 3)
xdata = np.transpose(xdata, (0, 1, 3, 2))
ydata = np.array(ydata)

np.save(os.path.join(output_dir, "xdata.npy"), xdata)
np.save(os.path.join(output_dir, "ydata.npy"), ydata)
pd.DataFrame({"EQ_ID": event_ids, "Mag": ydata}).to_csv(os.path.join(output_dir, "info_data_3stations.csv"), index=False)

print("Fichiers (3 stations) prêts pour l'entraînement dans", output_dir) 