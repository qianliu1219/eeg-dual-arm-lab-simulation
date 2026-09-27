"""
eeg_interface.py — EEG Interface (live signal + REAL robot control)

⚠️ À LANCER DANS UN TERMINAL ROS2 NORMAL (PAS eeg_venv) !
Ce script n'a plus besoin de tensorflow (les prédictions CNN ont déjà été
pré-calculées par eeg_prepare_robot_test_streams.py) — il peut donc tourner
dans l'environnement ROS2 normal et lancer/arrêter réellement le robot.

REQUIREMENTS (dans ce terminal normal) :
  pip install --user numpy matplotlib

SCENARIO — Play Concentration Signal :
  Le signal enregistré (court) alterne entre "repos" et "concentration".
  - Concentration détectée -> le robot démarre réellement (ou continue)
  - Repos détecté          -> le robot s'arrête réellement
  Comme le signal est court et ne couvre pas toute la durée de la mission,
  il est REJOUÉ EN BOUCLE depuis le début autant de fois que nécessaire,
  jusqu'à ce que le robot termine sa mission tout seul (ou soit stoppé
  par un clignement).

SCENARIO — Play Blink Signal :
  Rejoue le signal de clignement ; dès qu'un clignement est détecté,
  le robot est arrêté immédiatement (arrêt d'urgence).
"""
import pickle
import numpy as np
import subprocess
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

STREAMS_FILE = "eeg_robot_test_streams.pkl"
ROBOT_SCRIPT = os.path.expanduser(
    "~/colcon_ws_IGNITION/src/mybuddy_rl/dqn_gazebo_sequence_v3.py")

MIN_RUNTIME_BEFORE_STOP_SEC = 8   # ne jamais tuer le robot avant qu'il ait eu le temps de démarrer
DEBOUNCE_COUNT = 2                # nombre de détections consécutives nécessaires avant d'agir
CONC_INTERVAL_MS = 2000           # rythme d'analyse concentration : ~2s, comme la vraie durée d'un epoch enregistré
BLINK_INTERVAL_MS = 150           # rythme d'analyse clignement : rapide, pour un arrêt d'urgence réactif

# ---- Color palette ----
BG_COLOR = "#F8FAFC"
PANEL_COLOR = "#FFFFFF"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#64748B"
SIGNAL_LINE_COLOR = "#4F46E5"

COLOR_BLINK = "#3B82F6"; COLOR_BLINK_HOVER = "#2563EB"
COLOR_CONC = "#8B5CF6"; COLOR_CONC_HOVER = "#7C3AED"
COLOR_STOP = "#64748B"; COLOR_STOP_HOVER = "#475569"

with open(STREAMS_FILE, "rb") as f:
    streams = pickle.load(f)
X_conc_raw = streams['X_conc_raw']
y_pred_conc = streams['y_pred_conc']       # prédictions CNN déjà calculées
X_blink_raw = streams['X_blink_raw']
y_pred_blink = streams['y_pred_blink']     # prédictions CNN déjà calculées

print(f"✓ Flux chargés : {len(X_conc_raw)} fenêtres concentration, "
      f"{len(X_blink_raw)} fenêtres clignement (prédictions pré-calculées)")


class EEGInterface:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8), facecolor=BG_COLOR)
        self.fig.suptitle("EEG Interface — myBuddy Robot Control",
                           fontsize=18, fontweight='bold', color=TEXT_DARK, y=0.97)
        self.fig.text(0.5, 0.905, "Real EEG signal replay & live robot control",
                      fontsize=10.5, ha='center', color=TEXT_MUTED)

        self.ax_signal = self.fig.add_axes([0.09, 0.50, 0.83, 0.36])
        self.ax_signal.set_facecolor(PANEL_COLOR)
        self.signal_buffer = []
        self.line, = self.ax_signal.plot([], [], color=SIGNAL_LINE_COLOR, linewidth=1.4)
        self.ax_signal.set_xlim(0, 100)
        self.ax_signal.set_ylim(-3, 3)
        self.ax_signal.set_title("Live EEG Signal (real waveform)", fontsize=12,
                                  fontweight='bold', color=TEXT_DARK, loc='left')
        self.ax_signal.set_xlabel("Time (samples)", fontsize=9.5, color=TEXT_MUTED)
        self.ax_signal.grid(alpha=0.25)
        for spine in self.ax_signal.spines.values():
            spine.set_color('#E2E8F0')
        self.ax_signal.tick_params(colors=TEXT_MUTED, labelsize=8.5)

        self.ax_status = self.fig.add_axes([0.09, 0.40, 0.83, 0.07])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0.5, 0.5, "● STANDBY", fontsize=15, ha='center', va='center',
            fontweight='bold', color=TEXT_MUTED,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9', edgecolor='#CBD5E1'))

        self.ax_robot = self.fig.add_axes([0.09, 0.30, 0.83, 0.07])
        self.ax_robot.axis('off')
        self.robot_text = self.ax_robot.text(
            0.5, 0.5, "ROBOT: STOPPED", fontsize=14, ha='center', va='center',
            fontweight='bold', color='#B91C1C',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF2F2', edgecolor='#FCA5A5'))

        self.playing = False
        self.mode = None
        self.index = 0
        self.robot_process = None  # vrai processus du robot (subprocess)
        self.robot_started_at = None  # timestamp du dernier démarrage
        self.consec_rest = 0     # compteur anti-rebond (repos consécutifs)
        self.consec_conc = 0     # compteur anti-rebond (concentration consécutive)

        self._buttons = []
        self._add_button([0.20, 0.16, 0.27, 0.07], "▶  Play Blink Signal",
                          self.play_blink, COLOR_BLINK, COLOR_BLINK_HOVER)
        self._add_button([0.53, 0.16, 0.27, 0.07], "▶  Play Concentration Signal",
                          self.play_conc, COLOR_CONC, COLOR_CONC_HOVER)
        self._add_button([0.365, 0.06, 0.27, 0.07], "✕  Stop / Reset",
                          self.stop_all, COLOR_STOP, COLOR_STOP_HOVER)

        self.fig.text(0.5, 0.02,
                      "Real recorded EEG replay — robot actions are REAL (Gazebo subprocess)",
                      fontsize=8.5, ha='center', color=TEXT_MUTED, style='italic')

        self.timer = self.fig.canvas.new_timer(interval=BLINK_INTERVAL_MS)
        self.timer.add_callback(self.update)
        self.timer.start()

    def _set_timer_interval(self, interval_ms):
        """Recrée le minuteur avec un nouveau rythme (rapide pour le
        clignement, lent pour la concentration -> comme le vrai enregistrement)."""
        self.timer.stop()
        self.timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.update)
        self.timer.start()

    def _add_button(self, rect, label, callback, color, hover_color):
        ax = self.fig.add_axes(rect)
        btn = Button(ax, label, color=color, hovercolor=hover_color)
        btn.label.set_color('white')
        btn.label.set_fontweight('bold')
        btn.label.set_fontsize(10.5)
        btn.on_clicked(callback)
        self._buttons.append(btn)

    def _set_status(self, text, color, face, edge):
        self.status_text.set_text(text)
        self.status_text.set_color(color)
        self.status_text.get_bbox_patch().set_facecolor(face)
        self.status_text.get_bbox_patch().set_edgecolor(edge)

    def _set_robot(self, text, color, face, edge):
        self.robot_text.set_text(text)
        self.robot_text.set_color(color)
        self.robot_text.get_bbox_patch().set_facecolor(face)
        self.robot_text.get_bbox_patch().set_edgecolor(edge)

    def _start_robot(self):
        if self.robot_process is None or self.robot_process.poll() is not None:
            print("  🚀 Launching robot task...")
            self.robot_process = subprocess.Popen(["python3", ROBOT_SCRIPT])
            self.robot_started_at = time.time()

    def _stop_robot(self, force=False):
        if self.robot_process is not None and self.robot_process.poll() is None:
            elapsed = time.time() - (self.robot_started_at or 0)
            if not force and elapsed < MIN_RUNTIME_BEFORE_STOP_SEC:
                # Trop tôt : on laisse le robot le temps de vraiment démarrer/bouger
                return
            print("  🛑 Stopping robot task...")
            self.robot_process.terminate()
            try:
                self.robot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.robot_process.kill()
            self.robot_process = None

    def play_blink(self, event):
        self.mode = 'blink'
        self.index = 0
        self.playing = True
        self.signal_buffer = []
        self._set_timer_interval(BLINK_INTERVAL_MS)

    def play_conc(self, event):
        self.mode = 'conc'
        self.index = 0
        self.playing = True
        self.signal_buffer = []
        self.consec_conc = 0
        self.consec_rest = 0
        self._set_timer_interval(CONC_INTERVAL_MS)

    def stop_all(self, event):
        self.playing = False
        self._stop_robot()
        self._set_status("● STANDBY", TEXT_MUTED, '#F1F5F9', '#CBD5E1')
        self._set_robot("ROBOT: STOPPED", "#B91C1C", '#FEF2F2', '#FCA5A5')
        self.fig.canvas.draw_idle()

    def update(self):
        if not self.playing:
            return

        if self.mode == 'conc':
            # ---- Concentration : boucle le signal jusqu'à fin de mission ----
            if self.robot_process is not None and self.robot_process.poll() is not None:
                self.playing = False
                self._set_status("✓ MISSION COMPLETE", "#15803D", '#F0FDF4', '#86EFAC')
                self._set_robot("ROBOT: FINISHED", "#15803D", '#F0FDF4', '#86EFAC')
                self.robot_process = None
                self.fig.canvas.draw_idle()
                return

            i = self.index % len(X_conc_raw)
            epoch = X_conc_raw[i]
            pred = int(y_pred_conc[i])

            self.signal_buffer = epoch[0, :, 0].tolist()
            self.line.set_data(range(len(self.signal_buffer)), self.signal_buffer)
            self.ax_signal.set_xlim(0, len(self.signal_buffer))

            if pred == 1:
                self.consec_conc += 1
                self.consec_rest = 0
            else:
                self.consec_rest += 1
                self.consec_conc = 0

            if pred == 1:
                self._set_status("● CONCENTRATION DETECTED", "#059669", '#F0FDF4', '#86EFAC')
            else:
                self._set_status("● rest", TEXT_MUTED, '#F1F5F9', '#CBD5E1')

            # Anti-rebond : n'agit qu'après plusieurs détections consécutives
            if self.consec_conc >= DEBOUNCE_COUNT:
                self._start_robot()
                self._set_robot("ROBOT: RUNNING (beaker sequence)", "#15803D", '#F0FDF4', '#86EFAC')
            elif self.consec_rest >= DEBOUNCE_COUNT:
                self._stop_robot()
                if self.robot_process is None:
                    self._set_robot("ROBOT: STOPPED (rest)", "#B91C1C", '#FEF2F2', '#FCA5A5')

        else:
            # ---- Clignement : une seule détection suffit pour arrêter ----
            if self.index >= len(X_blink_raw):
                self.playing = False
                self.fig.canvas.draw_idle()
                return

            window = X_blink_raw[self.index]
            pred = int(y_pred_blink[self.index])

            self.signal_buffer = window[:, 0].tolist()
            self.line.set_data(range(len(self.signal_buffer)), self.signal_buffer)
            self.ax_signal.set_xlim(0, len(self.signal_buffer))

            if pred == 1:
                self._set_status("● BLINK DETECTED", "#DC2626", '#FEF2F2', '#FCA5A5')
                self._stop_robot(force=True)
                self._set_robot("ROBOT: STOPPED (blink detected)", "#B91C1C", '#FEF2F2', '#FCA5A5')
                self.playing = False
            else:
                self._set_status("● rest", TEXT_MUTED, '#F1F5F9', '#CBD5E1')

        self.index += 1
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = EEGInterface()
    plt.show()
