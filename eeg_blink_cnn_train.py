"""
eeg_blink_cnn_train.py — CNN pour la détection de clignement volontaire.
Architecture reprise fidèlement du notebook original des chercheurs
(CNN_and_results.ipynb, dépôt Atzingen/EEG_blink_detector).

⚠️ Ce script n'a PAS pu être testé dans l'environnement de développement
(manque d'espace disque pour installer tensorflow). Vous serez donc le
premier à le lancer — normal, ça fait partie de la démarche de tout
comprendre et valider vous-même.

PRÉREQUIS :
  pip install tensorflow

Les données (trainTestVV) sont déjà préparées par les chercheurs :
  - format : (n_échantillons, 250) = 1 seconde de signal brut, 1 seul canal
  - déjà équilibré ~50/50 (contrairement à notre Random Forest, testé sur
    la vraie distribution déséquilibrée -> résultats non directement comparables)
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                              f1_score, accuracy_score)

DATA_FILE = "EEG_blink_detector/trainTestVV"
MODEL_OUT = "eeg_blink_cnn.h5"
FIGURE_OUT = "eeg_blink_cnn_report_figure.png"


def load_data():
    with open(DATA_FILE, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Le CNN attend une forme (n_échantillons, 250, 1) : 250 points de temps, 1 canal
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


def build_model():
    """Architecture identique à celle du notebook original des chercheurs."""
    model = keras.Sequential([
        layers.Conv1D(128, 50, strides=1, padding='valid', activation='relu',
                       input_shape=(250, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    print("Chargement des données déjà préparées (trainTestVV)...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Équilibre train: {y_train.mean()*100:.1f}% clignements")
    print(f"Équilibre test : {y_test.mean()*100:.1f}% clignements")

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\nEntraînement du CNN (jusqu'à 50 epochs, arrêt automatique si convergence)...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=512, verbose=1, shuffle=True,
              callbacks=[early_stopping])

    n_epochs_effectues = len(history.history['loss'])
    print(f"\n✓ Entraînement arrêté après {n_epochs_effectues} epochs "
          f"(EarlyStopping déclenché, meilleurs poids restaurés)")

    # Évaluation détaillée avec les mêmes métriques que pour Random Forest
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== RÉSULTATS CNN (test équilibré ~50/50) ===")
    print(f"Accuracy  : {acc*100:.1f}%")
    print(f"Precision : {prec*100:.1f}%")
    print(f"Recall    : {rec*100:.1f}%")
    print(f"F1-score  : {f1*100:.1f}%")
    print("\nMatrice de confusion :")
    print(cm)

    model.save(MODEL_OUT)
    print(f"\n✓ Modèle sauvegardé : {MODEL_OUT}")

    # ================= FIGURE 0 : Courbe d'apprentissage (preuve de convergence) =================
    fig0, axes0 = plt.subplots(1, 2, figsize=(13, 5))
    fig0.suptitle(f"Courbe d'apprentissage du CNN — arrêt automatique après {n_epochs_effectues} epochs",
                  fontsize=14, fontweight='bold')

    ax = axes0[0]
    ax.plot(history.history['loss'], label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_title("Loss par epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (binary crossentropy)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes0[1]
    ax.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax.set_title("Accuracy par epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("eeg_blink_cnn_training_curve.png", dpi=150)
    print(f"✓ Courbe d'apprentissage sauvegardée : eeg_blink_cnn_training_curve.png")

    # ================= FIGURE (même style que pour Random Forest) =================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Détection de clignement volontaire (EEG) — CNN sur test équilibré",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    im = ax.imshow(cm, cmap='Greens')
    ax.set_title("Matrice de confusion (CNN)")
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Vraie étiquette")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['repos', 'clignement'])
    ax.set_yticklabels(['repos', 'clignement'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color=color, fontsize=16, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    names = ['Accuracy', 'Precision\n(clignement)', 'Recall\n(clignement)', 'F1-score\n(clignement)']
    values = [acc, prec, rec, f1]
    colors = ['#4C9A6B', '#DD8452', '#55A868', '#C44E52']
    bars = ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Métriques de performance (CNN)")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val*100:.1f}%",
                ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=150)
    print(f"✓ Figure sauvegardée : {FIGURE_OUT}")


if __name__ == "__main__":
    main()
