import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from obspy import read
from keras.models import load_model
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force le CPU

# === À MODIFIER : chemin vers le dossier contenant les fichiers SAC ===
sac_folder = "./subduction.000233"  # Exemple : "waveforms_data/waveforms/subduction.XXXXX"

try:
    # === Paramètres ===
    n_stations = 3
    n_composantes = 3
    n_temps = 181
    composantes = ["N", "E", "Z"]
    suffixes = [".LYN.sac", ".LYE.sac", ".LYZ.sac"]

    # Lister les fichiers SAC et extraire les noms de stations
    sac_files = [f for f in os.listdir(sac_folder) if f.endswith(".sac")]
    print("Fichiers SAC trouvés :", sac_files)
    stations = sorted(set(f.split(".")[0] for f in sac_files))
    print("Stations détectées :", stations)

    if len(stations) < n_stations:
        raise ValueError(f"Il faut au moins {n_stations} stations dans le dossier (trouvé : {len(stations)})")

    selected_stations = stations[:n_stations]
    print("Stations sélectionnées :", selected_stations)
    station_traces = []
    for station in selected_stations:
        comp_data = []
        for suffix in suffixes:
            sac_path = os.path.join(sac_folder, station + suffix)
            if os.path.exists(sac_path):
                try:
                    st = read(sac_path)
                    tr = st[0]
                    data = tr.data[:n_temps]
                    if len(data) < n_temps:
                        data = np.pad(data, (0, n_temps - len(data)), 'constant')
                except Exception as e:
                    print(f"Erreur lecture {sac_path}: {e}")
                    data = np.zeros(n_temps)
            else:
                print(f"Composante manquante : {sac_path}, padding avec des zéros.")
                data = np.zeros(n_temps)
            comp_data.append(data)
        station_traces.append(comp_data)  # [3, 181]

    xdata = np.stack(station_traces)  # [3, 3, 181]
    xdata = np.transpose(xdata, (0, 2, 1))  # [3, 181, 3]
    xdata = np.expand_dims(xdata, axis=0)  # [1, 3, 181, 3]

    # Charger le modèle
    model_path = "output/model/model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé à {model_path}")
    model = load_model(model_path, compile=False)

    # Prédire
    y_pred = model.predict(xdata)
    print(f"Magnitude prédite : {y_pred[0,0]:.2f}")

except Exception as e:
    print("Une erreur est survenue lors de la prédiction :")
    print(e) 