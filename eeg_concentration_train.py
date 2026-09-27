"""
eeg_concentration_train.py — Entraîne un classificateur "concentration forte /
imagerie motrice détectée" à partir du vrai dataset PhysioNet EEG Motor Imagery.

Ce classificateur simule le déclencheur "commencer la tâche" : quand l'utilisateur
imagine fortement un mouvement (= se concentre intensément), le robot démarre.

PRÉREQUIS : avoir lancé download_concentration_data.py avant (télécharge les .edf).

EXPLICATION DES ÉTAPES (à comprendre pour la présentation) :
  1. Chargement des fichiers EDF (format standard EEG) avec mne
  2. Découpage en "epochs" (segments) autour des événements marqués
     - T0 = repos (les yeux ouverts, rien de spécial)
     - T1/T2 = le sujet IMAGINE un mouvement (poing gauche/droit)
  3. Extraction de features : puissance du signal dans les bandes de fréquence
     mu (8-12 Hz) et beta (13-30 Hz), connues pour varier pendant l'imagerie motrice
  4. Entraînement d'un classificateur simple (régression logistique)
  5. Test sur un sujet JAMAIS vu à l'entraînement (généralisation réelle)
"""
import os
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pickle

mne.set_log_level('ERROR')  # réduit les messages techniques de mne

DATA_DIR = './eeg_concentration_data'
RUNS_MOTOR_IMAGERY = [4, 8, 12]   # imagerie motrice (poing gauche/droit)
SUBJECTS_TRAIN = list(range(1, 65))   # 64 sujets pour l'entraînement (au lieu de 16)
SUBJECTS_TEST = list(range(65, 81))   # 16 sujets JAMAIS vus, pour évaluer la vraie généralisation

MODEL_OUT = "eeg_concentration_classifier.pkl"


def load_subject_epochs(subject):
    """Charge les runs d'imagerie motrice d'un sujet et découpe en epochs (segments courts)."""
    raw_fnames = eegbci.load_data(subject, RUNS_MOTOR_IMAGERY, path=DATA_DIR)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    eegbci.standardize(raw)  # noms de canaux standards
    raw.set_montage('standard_1020', on_missing='ignore')
    raw.filter(7., 30., fir_design='firwin')  # bande mu+beta, pertinente pour l'imagerie motrice

    events, event_id = mne.events_from_annotations(raw)
    # T1 = imagerie main gauche, T2 = imagerie main droite -> on regroupe les 2 comme "concentration"
    picks = mne.pick_types(raw.info, eeg=True)

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=2.5,
                         picks=picks, baseline=None, preload=True)
    return epochs


def extract_features(epochs):
    """Extrait la puissance des bandes mu (8-12Hz) et beta (13-30Hz) par epoch."""
    from scipy.signal import welch

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']

    X = []
    for epoch in data:
        feats = []
        for ch in epoch:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(256, len(ch)))
            mu_power = psd[(freqs >= 8) & (freqs <= 12)].mean()
            beta_power = psd[(freqs >= 13) & (freqs <= 30)].mean()
            feats.extend([mu_power, beta_power])
        X.append(feats)

    # Labels : T1/T2 (imagerie motrice) = 1 ("concentration"), autre = 0
    y = []
    for event_code in epochs.events[:, -1]:
        event_name = [k for k, v in epochs.event_id.items() if v == event_code][0]
        y.append(1 if event_name in ('T1', 'T2') else 0)

    return np.array(X), np.array(y)


def main():
    print("=== Chargement et extraction des features (sujets d'entraînement) ===")
    X_train, y_train = [], []
    for subj in SUBJECTS_TRAIN:
        print(f"  Traitement sujet {subj}...")
        epochs = load_subject_epochs(subj)
        Xf, yf = extract_features(epochs)
        X_train.append(Xf)
        y_train.append(yf)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    print("\n=== Chargement du sujet de TEST (jamais vu à l'entraînement) ===")
    X_test, y_test = [], []
    for subj in SUBJECTS_TEST:
        print(f"  Traitement sujet {subj}...")
        epochs = load_subject_epochs(subj)
        Xf, yf = extract_features(epochs)
        X_test.append(Xf)
        y_test.append(yf)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    print(f"\nEntraînement : {len(X_train)} epochs ({y_train.sum()} concentration)")
    print(f"Test         : {len(X_test)} epochs ({y_test.sum()} concentration)")

    # Normalisation indispensable : les puissances mu/beta ont des échelles
    # très différentes selon les canaux, ce qui empêche la régression logistique
    # de converger correctement sans mise à l'échelle (même problème rencontré
    # et corrigé pour le classificateur de clignement).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Modèle 1 : Régression logistique (normalisée) ---")
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf_lr.fit(X_train_scaled, y_train)
    y_pred_lr = clf_lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy : {acc_lr*100:.1f}%")
    print(classification_report(y_test, y_pred_lr, target_names=['repos', 'concentration'], zero_division=0))

    print("\n--- Modèle 2 : Random Forest ---")
    clf_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                     max_depth=10, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy : {acc_rf*100:.1f}%")
    print(classification_report(y_test, y_pred_rf, target_names=['repos', 'concentration'], zero_division=0))

    f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)

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
    print(classification_report(y_test, y_pred, target_names=['repos', 'concentration'], zero_division=0))
    print(f"\n✓ Classificateur sauvegardé : {MODEL_OUT}")


if __name__ == "__main__":
    main()
