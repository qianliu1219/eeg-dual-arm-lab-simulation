"""
eeg_concentration_train_csp.py — Classification de l'imagerie motrice avec CSP
(Common Spatial Patterns), la méthode de référence historique pour ce type
de signal EEG, bien plus adaptée que le simple calcul de puissance mu/beta
par canal utilisé dans eeg_concentration_train.py.

Principe du CSP :
  Au lieu de regarder chaque canal séparément, le CSP trouve automatiquement
  les MEILLEURES COMBINAISONS LINÉAIRES de tous les canaux à la fois -
  celles qui séparent le mieux les deux classes (repos vs concentration).
  C'est une méthode de "filtrage spatial" spécifiquement conçue pour l'EEG
  moteur, utilisée depuis les années 2000 dans la quasi-totalité des
  systèmes BCI d'imagerie motrice.

PRÉREQUIS : avoir déjà lancé download_concentration_data.py.
"""
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              f1_score)
import pickle

from eeg_concentration_train import (SUBJECTS_TRAIN, SUBJECTS_TEST,
                                      load_subject_epochs)

mne.set_log_level('ERROR')

MODEL_OUT = "eeg_concentration_csp_classifier.pkl"
N_CSP_COMPONENTS = 6  # nombre de filtres spatiaux appris (valeur classique en BCI)


def load_raw_epochs(subjects):
    """Charge les epochs BRUTS (n_epochs, n_canaux, n_temps), sans calcul de
    features - le CSP a besoin du signal multi-canaux complet, pas de résumés."""
    X_all, y_all = [], []
    for subj in subjects:
        print(f"  Traitement sujet {subj}...")
        epochs = load_subject_epochs(subj)
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        y = []
        for event_code in epochs.events[:, -1]:
            event_name = [k for k, v in epochs.event_id.items() if v == event_code][0]
            y.append(1 if event_name in ('T1', 'T2') else 0)

        X_all.append(data)
        y_all.append(np.array(y))

    return np.concatenate(X_all), np.concatenate(y_all)


def main():
    print("=== Chargement des epochs BRUTS (sujets d'entraînement) ===")
    X_train, y_train = load_raw_epochs(SUBJECTS_TRAIN)

    print("\n=== Chargement des epochs BRUTS (sujets de TEST, jamais vus) ===")
    X_test, y_test = load_raw_epochs(SUBJECTS_TEST)

    print(f"\nEntraînement : {len(X_train)} epochs, forme {X_train.shape}")
    print(f"Test         : {len(X_test)} epochs, forme {X_test.shape}")

    # --- CSP : apprend les meilleurs filtres spatiaux sur les données d'entraînement ---
    print(f"\n--- Apprentissage des {N_CSP_COMPONENTS} filtres spatiaux CSP ---")
    csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    # --- Normalisation (comme pour les autres modèles) ---
    scaler = StandardScaler()
    X_train_csp_scaled = scaler.fit_transform(X_train_csp)
    X_test_csp_scaled = scaler.transform(X_test_csp)

    # --- Modèle 1 : LDA (l'appariement classique avec CSP dans la littérature BCI) ---
    print("\n--- Modèle 1 : CSP + LDA (pipeline classique BCI) ---")
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train_csp_scaled, y_train)
    y_pred_lda = clf_lda.predict(X_test_csp_scaled)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    print(f"Accuracy : {acc_lda*100:.1f}%")
    print(classification_report(y_test, y_pred_lda, target_names=['repos', 'concentration'], zero_division=0))

    # --- Modèle 2 : Régression logistique (pour comparaison) ---
    print("\n--- Modèle 2 : CSP + Régression logistique ---")
    clf_lr = LogisticRegression(class_weight='balanced', max_iter=2000)
    clf_lr.fit(X_train_csp_scaled, y_train)
    y_pred_lr = clf_lr.predict(X_test_csp_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy : {acc_lr*100:.1f}%")
    print(classification_report(y_test, y_pred_lr, target_names=['repos', 'concentration'], zero_division=0))

    f1_lda = f1_score(y_test, y_pred_lda, pos_label=1, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)

    if f1_lda >= f1_lr:
        print(f"\n✓ CSP + LDA retenu (F1={f1_lda:.2f} vs {f1_lr:.2f} pour LR)")
        clf, y_pred, acc = clf_lda, y_pred_lda, acc_lda
    else:
        print(f"\n✓ CSP + Régression logistique retenue (F1={f1_lr:.2f} vs {f1_lda:.2f} pour LDA)")
        clf, y_pred, acc = clf_lr, y_pred_lr, acc_lr

    with open(MODEL_OUT, "wb") as f:
        pickle.dump({'model': clf, 'csp': csp, 'scaler': scaler}, f)

    print(f"\n=== RÉSULTAT FINAL RETENU (CSP) ===")
    print(f"Accuracy : {acc*100:.1f}%")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=['repos', 'concentration'], zero_division=0))
    print(f"\n✓ Classificateur sauvegardé : {MODEL_OUT}")


if __name__ == "__main__":
    main()
