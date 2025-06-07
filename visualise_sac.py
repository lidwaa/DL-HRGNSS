from obspy import read
import matplotlib.pyplot as plt
import os

# Chemin vers le dossier de l'événement et préfixe de la station
# Exemple : dossier = "waveforms_data/waveforms/subduction.000001", station = "ACPM"
dossier = "waveforms_data/waveforms/subduction.000104"  # À adapter
station = "ANTC"  # À adapter

# Suffixes pour les composantes (à adapter selon la convention de nommage)
suffixes = {"N": ".LYN.sac", "E": ".LYE.sac", "Z": ".LYZ.sac"}

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, (comp, suffix) in enumerate(suffixes.items()):
    sac_path = os.path.join(dossier, station + suffix)
    if os.path.exists(sac_path):
        st = read(sac_path)
        tr = st[0]
        axs[i].plot(tr.times(), tr.data)
        axs[i].set_ylabel(f"Amplitude {comp}")
        axs[i].set_title(f"{station} - {comp}")
    else:
        axs[i].text(0.5, 0.5, f"Fichier manquant : {station + suffix}", ha='center', va='center')
        axs[i].set_ylabel(f"Amplitude {comp}")
        axs[i].set_title(f"{station} - {comp}")

axs[-1].set_xlabel("Temps (s)")
plt.tight_layout()
plt.show() 