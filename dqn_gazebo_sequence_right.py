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
from mybuddy_env_sequence_right import N_OBS, N_ACTIONS, STATES, ACTIONS

JOINTS_R = ['joint1_R', 'joint2_R', 'joint3_R', 'joint4_R', 'joint5_R', 'joint6_R']
GRIPPER_JOINTS = ['gripper_controller', 'L_gripper_controller']

# ============================================================
# POSITIONS CALIBRÉES EN IGNITION (validées manuellement)
# ============================================================
POSES_RIGHT = {
    'home'       : [0.0,       0.0,       0.0,       0.0,       0.0, 0.0],
    'waypoint1_R': [2.073456, -0.460768,  0.0,       0.0,       0.0, 0.0],
    'stirrer'    : [1.785476, -0.921536, -0.172788, -0.230384,  0.0, 0.0],
    'waypoint2_R': [2.131052, -1.439900, -0.172788, -0.230384,  0.0, 0.0],
    'mix'        : [1.785476, -1.727880, -0.172788, -0.057596,  0.0, 0.0],
}

GRIPPER_CLOSED_R = -0.377
GRIPPER_OPEN_R   = 0.0

MIX_JOINT2_LOW  = -1.55
MIX_JOINT2_HIGH = -1.90
MIX_CYCLES      = 4

MODEL_PATH = os.path.expanduser("~/colcon_ws_IGNITION/src/mybuddy_rl/best_dqn_sequence_right.pth")

ACTION_COLORS = {
    'go_home': '#888888',
    'go_waypoint1': '#FFA500',
    'go_stirrer': '#FF4444',
    'go_waypoint2': '#44AAFF',
    'go_mix': '#44FF44',
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
        print(f"✓ DQN chargé (bras droit : agitateur)")

    def select_action(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(s).squeeze().numpy()
            return self.q_net(s).argmax().item(), q_values


class Dashboard:
    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle("DQN myBuddy — Bras droit, agitateur (Ignition)",
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
            f"Agitateur : {'TENU' if state_dict['holding'] else 'LIBRE'}",
            f"Mélangé   : {'OUI' if state_dict['mixed'] else 'NON'}",
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


class IgnitionDemoRight(Node):
    def __init__(self):
        super().__init__('dqn_gazebo_sequence_right')
        self.right_arm = ActionClient(self, FollowJointTrajectory, '/right_arm_controller/follow_joint_trajectory')
        self.gripper = ActionClient(self, FollowJointTrajectory, '/gripper_action_controller/follow_joint_trajectory')
        self.right_arm.wait_for_server(timeout_sec=15.0)
        self.gripper.wait_for_server(timeout_sec=15.0)
        self.position = 'home'
        self.holding_stirrer = False
        self.mixed = False

    def attach_stirrer(self):
        try:
            subprocess.run(["ign", "topic", "-t", "/stirrer/attach",
                             "-m", "ignition.msgs.Empty", "-p", ""], timeout=3)
            print("  ✓ Agitateur attaché (Ignition)")
        except Exception as e:
            print(f"  ⚠ Erreur attach: {e}")

    def detach_stirrer(self):
        try:
            subprocess.run(["ign", "topic", "-t", "/stirrer/detach",
                             "-m", "ignition.msgs.Empty", "-p", ""], timeout=3)
            print("  ✓ Agitateur détaché (Ignition)")
        except Exception as e:
            print(f"  ⚠ Erreur detach: {e}")

    def move_to(self, positions, duration=4):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINTS_R
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj
        f = self.right_arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        if f.result():
            r = f.result().get_result_async()
            rclpy.spin_until_future_complete(self, r, timeout_sec=10.0)
        time.sleep(0.3)

    def set_gripper(self, value):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [value, 0.0]
        pt.velocities = [0.0, 0.0]
        pt.time_from_start = Duration(sec=2)
        traj.points = [pt]
        goal.trajectory = traj
        f = self.gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
        if f.result():
            r = f.result().get_result_async()
            rclpy.spin_until_future_complete(self, r, timeout_sec=5.0)
        time.sleep(0.5)

    def stir_motion(self, cycles=MIX_CYCLES):
        print("  → Mouvement de mélange...")
        base = POSES_RIGHT['mix']
        for c in range(cycles):
            low = list(base); low[1] = MIX_JOINT2_LOW
            high = list(base); high[1] = MIX_JOINT2_HIGH
            self.move_to(low, duration=1.0)
            self.move_to(high, duration=1.0)
        self.move_to(base, duration=1.0)
        print("  ✓ Mélange terminé")

    def get_state_dqn(self):
        pos_oh = np.zeros(5)
        pos_oh[STATES[self.position]] = 1.0
        return np.concatenate([pos_oh, [float(self.holding_stirrer)],
                                [float(self.mixed)]]).astype(np.float32)

    def execute(self, action):
        pos_name = list(STATES.keys())[action]
        self.move_to(POSES_RIGHT[pos_name])

        if pos_name == 'stirrer' and not self.holding_stirrer and not self.mixed:
            self.set_gripper(GRIPPER_CLOSED_R)
            self.attach_stirrer()
            self.holding_stirrer = True
            print("  ✓ AGITATEUR SAISI (attach) !")

        elif pos_name == 'mix' and self.holding_stirrer and not self.mixed:
            self.stir_motion()
            self.mixed = True
            print("  ✓ MÉLANGE EFFECTUÉ !")

        elif pos_name == 'stirrer' and self.holding_stirrer and self.mixed:
            self.detach_stirrer()
            self.holding_stirrer = False
            self.set_gripper(GRIPPER_OPEN_R)
            print("  ✓ AGITATEUR DÉTACHÉ — MISSION TERMINÉE ! 🎉")

        self.position = pos_name

    def run(self):
        print("\n  DÉMO DQN — BRAS DROIT, AGITATEUR (Ignition)\n")
        # Sécurité : détacher au cas où l'attache automatique aurait eu lieu au spawn
        self.detach_stirrer()

        agent = DQNAgent()
        dash = Dashboard()

        for step in range(15):
            state = self.get_state_dqn()
            action, q_vals = agent.select_action(state)
            name = ACTIONS[action]
            print(f"Step {step + 1} : {name}")
            sd = {'position': self.position, 'holding': self.holding_stirrer, 'mixed': self.mixed}
            dash.update(step + 1, sd, name, q_vals)
            self.execute(action)

            if self.position == 'stirrer' and self.mixed and not self.holding_stirrer:
                print("\n✅ MISSION ACCOMPLIE !")
                sd = {'position': self.position, 'holding': self.holding_stirrer, 'mixed': self.mixed}
                state = self.get_state_dqn()
                _, q_vals = agent.select_action(state)
                dash.update(step + 2, sd, 'SUCCÈS !', q_vals)
                break

        dash.save(os.path.expanduser("~/colcon_ws_IGNITION/src/mybuddy_rl/dqn_dashboard_right.png"))
        plt.close('all')  # se termine proprement au lieu de rester bloque


def main():
    rclpy.init()
    node = IgnitionDemoRight()
    time.sleep(2.0)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
