"""
dqn_train_sequence_v3.py — Entraînement DQN, séquence complète avec versement
5 positions, 5 actions, logique attach/pour/detach.

Génère 4 graphiques à la fin : reward, taux de succès, loss, epsilon
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from mybuddy_env_sequence_v3 import MyBuddyEnvSequenceV3, N_OBS, N_ACTIONS

EPISODES = 800
BATCH_SIZE = 32
GAMMA = 0.95
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 500
TARGET_UPDATE_EVERY = 5
BUFFER_SIZE = 5000

MODEL_OUT = "best_dqn_sequence_v3.pth"
CURVE_OUT = "training_curves_sequence_v3.png"


class QNetwork(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(ep):
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


def train():
    env = MyBuddyEnvSequenceV3()
    q_net = QNetwork(N_OBS, N_ACTIONS)
    target_net = QNetwork(N_OBS, N_ACTIONS)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    rewards_history = []
    success_history = []
    loss_history = []
    epsilon_history = []

    best_reward = -float('inf')

    for ep in range(EPISODES):
        obs, _ = env.reset()
        eps = epsilon_by_episode(ep)
        total_reward = 0.0
        episode_losses = []
        done = False
        truncated = False
        success = False

        while not (done or truncated):
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(obs).unsqueeze(0)
                    action = q_net(s).argmax().item()

            next_obs, reward, done, truncated, info = env.step(action)
            if info.get("success"):
                success = True
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                s, a, r, s2, d = buffer.sample(BATCH_SIZE)
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                s2 = torch.FloatTensor(s2)
                d = torch.FloatTensor(d)

                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(s2).max(1)[0]
                    target = r + GAMMA * max_next_q * (1 - d)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_losses.append(loss.item())

        rewards_history.append(total_reward)
        success_history.append(1.0 if success else 0.0)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)
        epsilon_history.append(eps)
        best_reward = max(best_reward, total_reward)

        if ep % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        if ep % 50 == 0 or ep == EPISODES - 1:
            avg_r = np.mean(rewards_history[-50:])
            avg_success = np.mean(success_history[-50:]) * 100
            avg_loss = np.mean(loss_history[-50:])
            print(f"Ep {ep:4d} | eps={eps:.3f} | reward={total_reward:+7.1f} | "
                  f"avg_reward50={avg_r:+7.1f} | succès50={avg_success:5.1f}% | loss={avg_loss:.4f}")

    torch.save(q_net.state_dict(), MODEL_OUT)
    print(f"\n✓ Entraînement terminé. Meilleur reward observé : {best_reward:.1f}")
    print(f"✓ Modèle final sauvegardé : {MODEL_OUT}")

    # ================================================================
    # GRAPHIQUES
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        window = 20
        episodes_x = np.arange(len(rewards_history))

        def smooth(data, w=window):
            if len(data) < w:
                return episodes_x, data
            sm = np.convolve(data, np.ones(w) / w, mode='valid')
            return np.arange(w - 1, len(data)), sm

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Entraînement DQN — Séquence complète avec versement (v3)",
                     fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(episodes_x, rewards_history, alpha=0.3, color='tab:blue', label='Reward brut')
        x_s, y_s = smooth(rewards_history)
        ax.plot(x_s, y_s, color='tab:blue', linewidth=2, label=f'Moyenne mobile ({window})')
        ax.set_title("Reward par épisode")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Reward total")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        x_s, y_s = smooth(success_history)
        ax.plot(x_s, np.array(y_s) * 100, color='tab:green', linewidth=2)
        ax.set_title(f"Taux de succès (moyenne mobile {window})")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Succès (%)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(episodes_x, loss_history, alpha=0.3, color='tab:red', label='Loss brute')
        x_s, y_s = smooth(loss_history)
        ax.plot(x_s, y_s, color='tab:red', linewidth=2, label=f'Moyenne mobile ({window})')
        ax.set_title("Loss (erreur du réseau)")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(episodes_x, epsilon_history, color='tab:orange', linewidth=2)
        ax.set_title("Epsilon (exploration → exploitation)")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Epsilon")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(CURVE_OUT, dpi=150)
        print(f"✓ Graphiques sauvegardés : {CURVE_OUT}")
    except Exception as e:
        print(f"⚠ Impossible de générer les graphiques : {e}")

    # ================================================================
    # Test final en mode greedy
    # ================================================================
    print("\n--- Test final (greedy, sans exploration) ---")
    q_net.load_state_dict(torch.load(MODEL_OUT))
    q_net.eval()
    obs, _ = env.reset()
    done = truncated = False
    steps = 0
    while not (done or truncated) and steps < 15:
        with torch.no_grad():
            s = torch.FloatTensor(obs).unsqueeze(0)
            action = q_net(s).argmax().item()
        obs, r, done, truncated, info = env.step(action)
        print(f"  Step {steps + 1}: {info['action']:15s} | reward={r:+6.1f}")
        steps += 1
    print("✅ SUCCÈS TEST FINAL" if done else "❌ ÉCHEC TEST FINAL")


if __name__ == "__main__":
    train()
