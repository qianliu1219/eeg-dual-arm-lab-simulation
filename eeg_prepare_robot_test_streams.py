"""
eeg_prepare_robot_test_streams.py — À LANCER DANS eeg_venv (celui avec mne/tensorflow).

Prépare à l'avance les données de test EEG (concentration + clignement) ET
PRÉ-CALCULE les prédictions des CNN, pour que l'interface finale n'ait plus
du tout besoin de tensorflow — elle pourra alors tourner dans un terminal
ROS2 normal et lancer/arrêter réellement le robot (rclpy), sans le conflit
protobuf rencontré précédemment.

Génère : eeg_robot_test_streams.pkl
  - X_conc_raw, y_conc_raw, y_pred_conc   : epochs bruts + PRÉDICTIONS CNN
  - X_blink_raw, y_blink_raw, y_pred_blink : fenêtres brutes + PRÉDICTIONS CNN
"""
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model

from eeg_concentration_train import SUBJECTS_TEST, load_subject_epochs, extract_features
from eeg_concentration_train_csp import load_raw_epochs
from eeg_concentration_train_cnn import normalize_epochs
from eeg_blink_train import DATA_DIR as BLINK_DATA_DIR, load_subject as load_blink_subject, extract_windows as extract_blink_windows

OUTPUT = "eeg_robot_test_streams.pkl"
BLINK_CNN_FILE = "eeg_blink_cnn.h5"
CONC_CNN_FILE = "eeg_concentration_cnn.h5"

BLINK_WINDOW_SAMPLES = 250   # 1 seconde à 250Hz (format attendu par le CNN clignement)
BLINK_LABEL_HALF_WIDTH = 0.4


def extract_raw_blink_windows(data_sig, blink_times):
    """Découpe le signal en fenêtres BRUTES de 250 échantillons (1s),
    format attendu par le CNN de clignement (eeg_blink_cnn.h5)."""
    ch1 = data_sig[:, 1]
    time = data_sig[:, 0]
    X, y = [], []
    n_windows = len(data_sig) // BLINK_WINDOW_SAMPLES
    for w in range(n_windows):
        start = w * BLINK_WINDOW_SAMPLES
        end = start + BLINK_WINDOW_SAMPLES
        seg = ch1[start:end].copy()
        # Normalisation par fenêtre (moyenne/écart-type), pour rester dans
        # une échelle raisonnable pour le réseau
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)

        t_center = time[start + BLINK_WINDOW_SAMPLES // 2]
        label = 0
        for bt in blink_times:
            if abs(t_center - bt) <= BLINK_LABEL_HALF_WIDTH:
                label = 1
                break
        X.append(seg)
        y.append(label)
    return np.array(X), np.array(y)


def main():
    print("Chargement des modèles CNN (pour pré-calculer les prédictions)...")
    blink_cnn = load_model(BLINK_CNN_FILE)
    conc_cnn = load_model(CONC_CNN_FILE)

    print("\n=== Préparation du flux de test CONCENTRATION (brut + prédictions) ===")
    test_subject = SUBJECTS_TEST[0]
    print(f"Sujet utilisé : {test_subject}")

    X_conc_raw, y_conc_raw = load_raw_epochs([test_subject])
    X_conc_raw = normalize_epochs(X_conc_raw)
    n_channels, n_samples = X_conc_raw.shape[1], X_conc_raw.shape[2]
    X_conc_raw = X_conc_raw.reshape(-1, n_channels, n_samples, 1)
    print(f"→ {len(X_conc_raw)} epochs BRUTS préparés (forme {X_conc_raw.shape})")

    proba_conc = conc_cnn.predict(X_conc_raw, verbose=0).flatten()
    y_pred_conc = (proba_conc > 0.5).astype(int)
    print(f"→ Prédictions pré-calculées : {y_pred_conc.sum()} concentration / {len(y_pred_conc)} fenêtres")

    print("\n=== Préparation du flux de test CLIGNEMENT (brut + prédictions) ===")
    blink_files = sorted([f for f in os.listdir(BLINK_DATA_DIR) if '_data' in f])
    test_file = blink_files[10]
    print(f"Sujet utilisé : {test_file}")
    sig, blink_times = load_blink_subject(BLINK_DATA_DIR, test_file)

    X_blink_raw, y_blink_raw = extract_raw_blink_windows(sig, blink_times)
    X_blink_raw = X_blink_raw.reshape(-1, BLINK_WINDOW_SAMPLES, 1)
    print(f"→ {len(X_blink_raw)} fenêtres BRUTES préparées (forme {X_blink_raw.shape})")

    proba_blink = blink_cnn.predict(X_blink_raw, verbose=0).flatten()
    y_pred_blink = (proba_blink > 0.5).astype(int)
    print(f"→ Prédictions pré-calculées : {y_pred_blink.sum()} clignements / {len(y_pred_blink)} fenêtres")

    with open(OUTPUT, "wb") as f:
        pickle.dump({
            'X_conc_raw': X_conc_raw, 'y_conc_raw': y_conc_raw, 'y_pred_conc': y_pred_conc,
            'X_blink_raw': X_blink_raw, 'y_blink_raw': y_blink_raw, 'y_pred_blink': y_pred_blink,
        }, f)

    print(f"\n✓ Flux de test + prédictions sauvegardés dans : {OUTPUT}")
    print("  Copiez ce fichier vers votre workspace ROS2 pour l'étape suivante :")
    print("  cp eeg_robot_test_streams.pkl ~/colcon_ws_IGNITION/src/mybuddy_rl/")


if __name__ == "__main__":
    main()
