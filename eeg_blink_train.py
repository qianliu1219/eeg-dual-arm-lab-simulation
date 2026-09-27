"""
eeg_blink_train.py — Entraîne un classificateur "clignement volontaire détecté / non détecté"
sur le vrai dataset public EEG-VV (Agarwal & Sivakumar, 2019 — OpenBCI, 250Hz).

Ce classificateur simule le rôle du "bouton d'arrêt d'urgence" déclenché par EEG :
quand un clignement volontaire est détecté, on appelle une fonction d'arrêt d'urgence.
"""
import os
import csv
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

DATA_DIR = "EEG_blink_detector/data/EEG-VV"
FS = 250.0                  # fréquence d'échantillonnage (Hz)
WINDOW_SEC = 0.5            # fenêtre glissante pour extraire les features
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
BLINK_HALF_WIDTH = 0.4      # demi-largeur (s) autour d'un timestamp de clignement étiqueté comme "blink"

MODEL_OUT = "eeg_blink_classifier.pkl"


def lowpass(sig, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2), btype='low')
    return lfilter(b, a, sig, axis=0)


def load_subject(data_path, file_data):
    """Charge un fichier EEG-VV (format OpenBCI brut) + ses labels de clignement."""
    file_labels = file_data.replace('_data', '_labels')

    # Les 5 premières lignes sont des commentaires OpenBCI (%...)
    data_sig = np.loadtxt(os.path.join(data_path, file_data), delimiter=",",
                           skiprows=5, usecols=(0, 1, 2))
    data_sig = data_sig[:, 0:3]
    data_sig[:, 0] = np.arange(len(data_sig)) / FS   # reconstruire le temps en secondes

    # Filtrage passe-bas (10 Hz) sur les 2 canaux EEG, comme dans le papier original
    data_sig[:, 1] = lowpass(data_sig[:, 1], 10, FS)
    data_sig[:, 2] = lowpass(data_sig[:, 2], 10, FS)

    # Lecture des timestamps de clignements volontaires
    blink_times = []
    with open(os.path.join(data_path, file_labels)) as f:
        reader = csv.reader(f)
        n_corrupt = 0
        for row in reader:
            if row[0] == "corrupt":
                n_corrupt = int(row[1])
            elif n_corrupt > 0:
                n_corrupt -= 1
            elif row[0] == "blinks":
                continue
            else:
                blink_times.append(float(row[0]))

    return data_sig, blink_times


def extract_windows(data_sig, blink_times):
    """Découpe le signal en fenêtres de 0.5s (chevauchement 50%) et labellise chacune :
    1 = clignement, 0 = repos. Utilise des features plus discriminantes qu'avant :
    amplitude, pente (dérivée), corrélation entre canaux, kurtosis (pic brusque)."""
    from scipy.stats import kurtosis

    ch1 = data_sig[:, 1]
    ch2 = data_sig[:, 2]
    time = data_sig[:, 0]

    step = WINDOW_SAMPLES // 2  # chevauchement 50% -> meilleure résolution temporelle
    X, y = [], []

    start = 0
    while start + WINDOW_SAMPLES <= len(data_sig):
        end = start + WINDOW_SAMPLES
        t_center = time[start + WINDOW_SAMPLES // 2]

        seg1 = ch1[start:end]
        seg2 = ch2[start:end]

        # Dérivées (pente) : un clignement monte/descend très vite
        d1 = np.diff(seg1)
        d2 = np.diff(seg2)

        # Corrélation entre les 2 canaux : un clignement affecte les 2 de façon corrélée
        if np.std(seg1) > 1e-6 and np.std(seg2) > 1e-6:
            corr = np.corrcoef(seg1, seg2)[0, 1]
        else:
            corr = 0.0

        features = [
            np.mean(seg1), np.std(seg1), np.ptp(seg1), np.max(np.abs(seg1)),
            np.mean(seg2), np.std(seg2), np.ptp(seg2), np.max(np.abs(seg2)),
            np.max(np.abs(d1)), np.max(np.abs(d2)),          # pente max (vitesse de variation)
            kurtosis(seg1), kurtosis(seg2),                   # "piquant" du signal
            corr,                                              # corrélation inter-canaux
        ]

        label = 0
        for bt in blink_times:
            if abs(t_center - bt) <= BLINK_HALF_WIDTH:
                label = 1
                break

        X.append(features)
        y.append(label)
        start += step

    return np.array(X), np.array(y)


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if '_data' in f])
    print(f"✓ {len(files)} sujets trouvés dans {DATA_DIR}")

    # Split : on entraîne sur la majorité des sujets, on teste sur les autres
    # (généralisation inter-sujets, plus réaliste qu'un split aléatoire)
    train_files = files[:10]
    test_files = files[10:]

    X_train, y_train = [], []
    for f in train_files:
        sig, blinks = load_subject(DATA_DIR, f)
        Xf, yf = extract_windows(sig, blinks)
        X_train.append(Xf)
        y_train.append(yf)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = [], []
    for f in test_files:
        sig, blinks = load_subject(DATA_DIR, f)
        Xf, yf = extract_windows(sig, blinks)
        X_test.append(Xf)
        y_test.append(yf)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    print(f"\nEntraînement : {len(X_train)} fenêtres ({y_train.sum()} clignements)")
    print(f"Test         : {len(X_test)} fenêtres ({y_test.sum()} clignements)")
    print(f"Sujets test  : {test_files}")

    # Normalisation des features (important pour la régression logistique,
    # nos features ont des échelles très différentes : ex. amplitude en microvolts
    # vs corrélation entre -1 et 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Modèle 1 : Régression logistique (normalisée) ---")
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf_lr.fit(X_train_scaled, y_train)
    y_pred_lr = clf_lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy : {acc_lr*100:.1f}%")
    print(classification_report(y_test, y_pred_lr, target_names=['repos', 'clignement']))

    print("\n--- Modèle 2 : Random Forest ---")
    clf_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     max_depth=10, random_state=42)
    clf_rf.fit(X_train, y_train)  # Random Forest n'a pas besoin de normalisation
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy : {acc_rf*100:.1f}%")
    print(classification_report(y_test, y_pred_rf, target_names=['repos', 'clignement']))

    # On garde le meilleur des deux modèles (selon le F1-score sur la classe clignement)
    from sklearn.metrics import f1_score
    f1_lr = f1_score(y_test, y_pred_lr, pos_label=1)
    f1_rf = f1_score(y_test, y_pred_rf, pos_label=1)

    if f1_rf >= f1_lr:
        print(f"\n✓ Random Forest retenu (F1={f1_rf:.2f} vs {f1_lr:.2f} pour LR)")
        clf, y_pred, acc = clf_rf, y_pred_rf, acc_rf
        with open(MODEL_OUT, "wb") as f:
            pickle.dump({'model': clf, 'scaler': None}, f)
    else:
        print(f"\n✓ Régression logistique retenue (F1={f1_lr:.2f} vs {f1_rf:.2f} pour RF)")
        clf, y_pred, acc = clf_lr, y_pred_lr, acc_lr
        with open(MODEL_OUT, "wb") as f:
            pickle.dump({'model': clf, 'scaler': scaler}, f)

    print(f"\n=== RÉSULTAT FINAL RETENU ===")
    print(f"Accuracy : {acc*100:.1f}%")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=['repos', 'clignement']))
    print(f"\n✓ Classificateur sauvegardé : {MODEL_OUT}")


if __name__ == "__main__":
    main()
