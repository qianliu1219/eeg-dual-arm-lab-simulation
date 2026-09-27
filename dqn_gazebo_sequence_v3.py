import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import torch
import torch.nn as nn
import numpy as np
import time
import subprocess
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mybuddy_env_sequence_v3 import N_OBS, N_ACTIONS, STATES, ACTIONS

JOINTS_L = ['joint1_L', 'joint2_L', 'joint3_L', 'joint4_L', 'joint5_L', 'joint6_L']

# ============================================================
# POSITIONS CALIBRÉES EN IGNITION (validées manuellement via
# rqt_joint_trajectory_controller)
# ============================================================
POSES_LEFT = {
    'home'      : [0.0,       0.0,      0.0,       0.0,       0.0, 0.0],
    'waypoint1' : [-2.073456, 0.460768, 0.0,       0.0,       0.0, 0.0],
    'beaker'    : [-1.785476, 0.863940, 0.287980,  0.115192,  0.0, 0.244344],
    'waypoint2' : [-2.361436, 1.209516, 0.287980,  0.115192,  0.0, 0.244344],
    'pre_pour'  : [-2.073456, 1.324708, 0.403172, -0.115192,  0.0, 1.710408],
}

MODEL_PATH = os.path.expanduser("~/colcon_ws_IGNITION/src/mybuddy_rl/best_dqn_sequence_v3.pth")

ACTION_COLORS = {
    'go_home': '#888888',
    'go_waypoint1': '#FFA500',
    'go_beaker': '#FF4444',
    'go_waypoint2': '#44AAFF',
    'go_pre_pour': '#44FF44',
}


class QNetwork(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork(N_OBS, N_ACTIONS)
        self.q_net.load_state_dict(torch.load(MODEL_PATH))
        self.q_net.eval()
        print(f"✓ DQN chargé (v3 : waypoints + versement)")

    def select_action(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(s).squeeze().numpy()
            return self.q_net(s).argmax().item(), q_values


class Dashboard:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle("DQN myBuddy — Séquence complète avec versement (Ignition)",
                           fontsize=13, fontweight='bold')
        self.history = []
        plt.show(block=False)
        plt.pause(0.5)

    def update(self, step, state_dict, action_name, q_values):
        self.history.append(action_name)
        ax = self.axes[0]
        ax.clear()
        ax.set_title(f"Étape {step}", fontweight='bold')
        info = [
            f"Position : {state_dict['position']}",
            f"Bécher    : {'TENU' if state_dict['holding'] else 'LIBRE'}",
            f"Versé     : {'OUI' if state_dict['poured'] else 'NON'}",
        ]
        for i, txt in enumerate(info):
            ax.text(0.05, 0.8 - i * 0.2, txt, fontsize=12, transform=ax.transAxes)
        ax.text(0.05, 0.15, "→ Action :", fontsize=11, transform=ax.transAxes, fontweight='bold')
        ax.text(0.05, 0.02, action_name, fontsize=13, transform=ax.transAxes,
                color=ACTION_COLORS.get(action_name, 'black'), fontweight='bold')
        ax.axis('off')

        ax = self.axes[1]
        ax.clear()
        ax.set_title("Q-values (préférence DQN)", fontweight='bold')
        actions = list(ACTIONS.values())
        colors = [ACTION_COLORS[a] for a in actions]
        bars = ax.barh(actions, q_values, color=colors)
        best = np.argmax(q_values)
        bars[best].set_edgecolor('red')
        bars[best].set_linewidth(3)
        ax.grid(alpha=0.3, axis='x')

        ax = self.axes[2]
        ax.clear()
        ax.set_title("Séquence d'actions", fontweight='bold')
        for i, act in enumerate(self.history):
            ax.barh(i, 1, color=ACTION_COLORS.get(act, 'gray'), edgecolor='black')
            ax.text(0.5, i, f"{i + 1}. {act}", ha='center', va='center', fontweight='bold', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, max(len(self.history), 8) - 0.5)
        ax.invert_yaxis()
        ax.axis('off')

        plt.tight_layout()
        plt.pause(0.3)

    def save(self, path):
        plt.savefig(path, dpi=150)
        print(f"Dashboard : {path}")

    def keep_open(self):
        plt.ioff()
        plt.show()


class IgnitionDemoV3(Node):
    def __init__(self):
        super().__init__('dqn_gazebo_sequence_v3')
        self.left_arm = ActionClient(self, FollowJointTrajectory, '/left_arm_controller/follow_joint_trajectory')
        self.left_arm.wait_for_server(timeout_sec=15.0)
        self.position = 'home'
        self.holding_beaker = False
        self.poured = False

    def attach_beaker(self):
        try:
            subprocess.run(
                ["ign", "topic", "-t", "/beaker/attach",
                 "-m", "ignition.msgs.Empty", "-p", ""],
                timeout=3
            )
            print("  ✓ Bécher attaché (Ignition)")
        except Exception as e:
            print(f"  ⚠ Erreur attach: {e}")

    def detach_beaker(self):
        try:
            subprocess.run(
                ["ign", "topic", "-t", "/beaker/detach",
                 "-m", "ignition.msgs.Empty", "-p", ""],
                timeout=3
            )
            print("  ✓ Bécher détaché (Ignition)")
        except Exception as e:
            print(f"  ⚠ Erreur detach: {e}")

    def move_to(self, pos_name):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINTS_L
        pt = JointTrajectoryPoint()
        pt.positions = POSES_LEFT[pos_name]
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=4)
        traj.points = [pt]
        goal.trajectory = traj
        f = self.left_arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        if f.result():
            r = f.result().get_result_async()
            rclpy.spin_until_future_complete(self, r, timeout_sec=10.0)
        time.sleep(0.3)

    def get_state_dqn(self):
        pos_oh = np.zeros(5)
        pos_oh[STATES[self.position]] = 1.0
        return np.concatenate([pos_oh, [float(self.holding_beaker)],
                                [float(self.poured)]]).astype(np.float32)

    def execute(self, action):
        pos_name = list(STATES.keys())[action]
        self.move_to(pos_name)

        if pos_name == 'beaker' and not self.holding_beaker and not self.poured:
            self.attach_beaker()
            self.holding_beaker = True
            print("  ✓ BÉCHER SAISI (attach) !")

        elif pos_name == 'pre_pour' and self.holding_beaker and not self.poured:
            self.poured = True
            print("  ✓ VERSEMENT EFFECTUÉ (bécher toujours attaché) !")

        elif pos_name == 'beaker' and self.holding_beaker and self.poured:
            self.detach_beaker()
            self.holding_beaker = False
            print("  ✓ BÉCHER DÉTACHÉ — MISSION TERMINÉE ! 🎉")

        self.position = pos_name

    def run(self):
        print("\n  DÉMO DQN v3 — SÉQUENCE COMPLÈTE AVEC VERSEMENT (Ignition)\n")
        agent = DQNAgent()
        dash = Dashboard()

        for step in range(15):
            state = self.get_state_dqn()
            action, q_vals = agent.select_action(state)
            name = ACTIONS[action]
            print(f"Step {step + 1} : {name}")
            sd = {'position': self.position, 'holding': self.holding_beaker, 'poured': self.poured}
            dash.update(step + 1, sd, name, q_vals)
            self.execute(action)

            if self.position == 'beaker' and self.poured and not self.holding_beaker:
                print("\n✅ MISSION ACCOMPLIE !")
                sd = {'position': self.position, 'holding': self.holding_beaker, 'poured': self.poured}
                state = self.get_state_dqn()
                _, q_vals = agent.select_action(state)
                dash.update(step + 2, sd, 'SUCCÈS !', q_vals)
                break

        dash.save(os.path.expanduser("~/colcon_ws_IGNITION/src/mybuddy_rl/dqn_dashboard_v3.png"))
        plt.close('all')  # se termine proprement au lieu de rester bloque


def main():
    rclpy.init()
    node = IgnitionDemoV3()
    time.sleep(2.0)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
