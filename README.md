# DL-HRGNSS — Prédiction de la magnitude sismique à partir de données GNSS

Ce projet permet d'entraîner et d'utiliser un modèle de deep learning pour estimer la magnitude d'un séisme à partir des données GNSS (fichiers SAC) de plusieurs stations.

---

## 1. Installation & Dépendances

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, ObsPy, tqdm, matplotlib

Installe tout avec :

```bash
pip install -r requirements.txt
```

---

## 2. Organisation des données

- **waveforms_data/waveforms/subduction.XXXXX/** :  
  Dossiers d'événements, chacun contenant les fichiers SAC des stations (ex : `STATION.LYN.sac`, `STATION.LYE.sac`, `STATION.LYZ.sac`).

- **data/GNSS_M3S_181/** :  
  Dossier où sont générés les fichiers d'entraînement :

  - `xdata.npy` : données GNSS (shape : n_events, 3, 181, 3)
  - `ydata.npy` : magnitudes cibles
  - `info_data_3stations.csv` : métadonnées

- **output/model/model.h5** :  
  Modèle entraîné sauvegardé.

---

## 3. Préparation des données

Pour générer les fichiers d'entraînement à partir de tous les événements :

```bash
python prepare_gnss_3stations.py
```

- Le script sélectionne 3 stations par événement, lit les 3 composantes (N, E, Z) sur 181 points, et gère les fichiers manquants.
- Les fichiers sont générés dans `data/GNSS_M3S_181/`.

---

## 4. Entraînement du modèle

Lance l'entraînement avec :

```bash
python main.py
```

- Le script utilise les fichiers `xdata.npy` et `ydata.npy` pour entraîner un modèle CNN.
- Le modèle est sauvegardé dans `output/model/model.h5`.

---

## 5. Prédiction sur de nouveaux événements

Pour prédire la magnitude à partir d'un dossier de 9 fichiers SAC (3 stations × 3 composantes) :

1. Modifie la variable `sac_folder` dans `predict_magnitude_from_sac_folder.py` pour pointer vers ton dossier.
2. Lance :
   ```bash
   python predict_magnitude_from_sac_folder.py
   ```

- Le script affiche la magnitude prédite.

Pour prédire sur un lot d'événements, prépare un `xdata.npy` et utilise un mini-script :

```python
import numpy as np
from keras.models import load_model
xdata = np.load("data/GNSS_M3S_181/xdata.npy")
model = load_model("output/model/model.h5", compile=False)
y_pred = model.predict(xdata)
print(y_pred.flatten())
```

---

## 6. Visualisation

- Utilise `visualise_sac.py` pour afficher les séries temporelles d'une station (3 composantes).
- Utilise `Data_plot.ipynb` pour explorer les données GNSS ou les résultats.

---

## 7. Bonnes pratiques

- Le dossier `waveforms_data/` est ignoré par git (voir `.gitignore`).
- Les gros fichiers de données et modèles ne sont pas versionnés.

---

## 8. Ressources complémentaires

- Données GNSS originales : [Zenodo 10.5281/zenodo.4008690](https://doi.org/10.5281/zenodo.4008690)
- Article de référence : [Quinteros et al., 2024](https://doi.org/10.1016/j.jsames.2024.104815)

---

**Pour toute question ou adaptation de script, consulte les fichiers Python du projet ou demande de l'aide !**
