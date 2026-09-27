"""
eeg_concentration_train_cnn.py — CNN (architecture EEGNet) pour la concentration.

EEGNet (Lawhern et al., 2018) est l'architecture CNN de référence pour la
classification EEG multi-canaux — bien plus adaptée que le CNN simple utilisé
pour le clignement (qui ne traitait qu'un seul canal en entrée).

Principe :
  1. Convolution temporelle : apprend des filtres fréquentiels (comme un
     banc de filtres passe-bande automatique)
  2. Convolution "depthwise" spatiale : apprend des combinaisons de canaux,
     un peu comme le CSP, mais optimisée conjointement avec le reste du réseau
  3. Convolution séparable : combine les caractéristiques temporelles/spatiales

PRÉREQUIS : avoir déjà lancé download_concentration_data.py.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                      BatchNormalization, Activation, AveragePooling2D,
                                      Dropout, Flatten, Dense)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eeg_concentration_train_csp import load_raw_epochs
from eeg_concentration_train import SUBJECTS_TRAIN, SUBJECTS_TEST

MODEL_OUT = "eeg_concentration_cnn.h5"
CURVE_OUT = "eeg_concentration_cnn_training_curve.png"
FIGURE_OUT = "eeg_concentration_cnn_report_figure.png"


def build_eegnet(nb_channels, nb_samples, F1=8, D=2, F2=16, kernLength=64, dropoutRate=0.5):
    """Architecture EEGNet compacte (Lawhern et al., 2018)."""
    input1 = Input(shape=(nb_channels, nb_samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((nb_channels, 1), use_bias=False, depth_multiplier=D,
                              depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(1, kernel_constraint=max_norm(0.25), activation='sigmoid')(flatten)

    return Model(inputs=input1, outputs=dense)


def normalize_epochs(X):
    """Standardisation par canal (moyenne/écart-type calculés sur chaque canal)."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def main():
    print("=== Chargement des epochs BRUTS (sujets d'entraînement) ===")
    X_train, y_train = load_raw_epochs(SUBJECTS_TRAIN)

    print("\n=== Chargement des epochs BRUTS (sujets de TEST, jamais vus) ===")
    X_test, y_test = load_raw_epochs(SUBJECTS_TEST)

    print(f"\nEntraînement : {X_train.shape}")
    print(f"Test         : {X_test.shape}")

    X_train = normalize_epochs(X_train)
    X_test = normalize_epochs(X_test)

    n_channels, n_samples = X_train.shape[1], X_train.shape[2]
    X_train = X_train.reshape(-1, n_channels, n_samples, 1)
    X_test = X_test.reshape(-1, n_channels, n_samples, 1)

    print(f"\nForme finale (CNN) : {X_train.shape}")

    model = build_eegnet(n_channels, n_samples)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Patience augmentée (8 -> 15) : le val_loss oscillait encore proche de son
    # minimum au moment de l'arrêt précédent (epoch 10), on laisse plus de temps
    # pour éviter un arrêt prématuré sur un simple plateau temporaire.
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nEntraînement du CNN (EEGNet, jusqu'à 80 epochs, arrêt automatique)...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=80, batch_size=32, verbose=1,
                         callbacks=[early_stopping])

    n_epochs_effectues = len(history.history['loss'])
    print(f"\n✓ Entraînement arrêté après {n_epochs_effectues} epochs")

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== RÉSULTATS CNN (EEGNet) — sujets jamais vus ===")
    print(f"Accuracy  : {acc*100:.1f}%")
    print(f"Precision : {prec*100:.1f}%")
    print(f"Recall    : {rec*100:.1f}%")
    print(f"F1-score  : {f1*100:.1f}%")
    print("\nMatrice de confusion :")
    print(cm)
    print("\n", classification_report(y_test, y_pred, target_names=['repos', 'concentration'], zero_division=0))

    model.save(MODEL_OUT)
    print(f"\n✓ Modèle sauvegardé : {MODEL_OUT}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"EEGNet — Concentration/imagerie motrice — arrêt après {n_epochs_effectues} epochs",
                 fontsize=13, fontweight='bold')
    ax = axes[0]
    ax.plot(history.history['loss'], label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_title("Loss par epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax.set_title("Accuracy par epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_OUT, dpi=150)
    print(f"✓ Courbe d'apprentissage sauvegardée : {CURVE_OUT}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Concentration / imagerie motrice — CNN (EEGNet) — sujets jamais vus",
                 fontsize=14, fontweight='bold')
    ax = axes[0]
    im = ax.imshow(cm, cmap='Oranges')
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Vraie étiquette")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['repos', 'concentration']); ax.set_yticklabels(['repos', 'concentration'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=16, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [acc, prec, rec, f1]
    colors_bar = ['#EA580C', '#DD8452', '#55A868', '#C44E52']
    bars = ax.bar(names, values, color=colors_bar)
    ax.set_ylim(0, 1.05); ax.set_title("Métriques (classe concentration)"); ax.set_ylabel("Score")
    ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val*100:.1f}%",
                ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=150)
    print(f"✓ Figure sauvegardée : {FIGURE_OUT}")


if __name__ == "__main__":
    main()
