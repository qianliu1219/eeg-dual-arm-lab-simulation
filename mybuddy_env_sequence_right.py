"""
mybuddy_env_sequence_right.py — DQN bras droit (agitateur)
Positions : home, waypoint1_R, stirrer, waypoint2_R, mix

Logique :
  - 1ère arrivée à 'stirrer'  -> ATTACHE (saisie de l'agitateur)
  - arrivée à 'mix'           -> MÉLANGE (reste attaché)
  - 2e arrivée à 'stirrer'    -> DÉTACHE (fin de mission)
"""
import gymnasium as gym
import numpy as np

STATES = {
    'home'       : 0,
    'waypoint1_R': 1,
    'stirrer'    : 2,
    'waypoint2_R': 3,
    'mix'        : 4,
}
N_POSITIONS = len(STATES)

ACTIONS = {
    0: 'go_home',
    1: 'go_waypoint1',
    2: 'go_stirrer',
    3: 'go_waypoint2',
    4: 'go_mix',
}
N_ACTIONS = len(ACTIONS)
N_OBS = N_POSITIONS + 2  # one-hot position + holding_stirrer + mixed

# Transitions autorisées : uniquement entre positions adjacentes
SAFE_TRANSITIONS = {
    'home'       : ['home', 'waypoint1_R'],
    'waypoint1_R': ['waypoint1_R', 'home', 'stirrer'],
    'stirrer'    : ['stirrer', 'waypoint1_R', 'waypoint2_R'],
    'waypoint2_R': ['waypoint2_R', 'stirrer', 'mix'],
    'mix'        : ['mix', 'waypoint2_R'],
}

REWARD_SAISIE       =   5.0
REWARD_MELANGE      =  20.0
REWARD_DEPOT        = 100.0
REWARD_STEP_PENALTY =  -1.0
REWARD_UNSAFE       = -20.0
MAX_STEPS           =  15


class MyBuddyEnvSequenceRight(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.position = 'home'
        self.holding_stirrer = False
        self.mixed = False
        self.step_count = 0
        self.done = False
        print(f"MyBuddyEnvSequenceRight (5 positions, attach/mélange/detach) ✓")

    def _get_obs(self):
        pos_one_hot = np.zeros(N_POSITIONS)
        pos_one_hot[STATES[self.position]] = 1.0
        return np.concatenate([
            pos_one_hot,
            [float(self.holding_stirrer)],
            [float(self.mixed)],
        ]).astype(np.float32)

    def step(self, action):
        action_name = ACTIONS[action]
        reward = REWARD_STEP_PENALTY
        info = {"action": action_name, "success": False}

        new_pos = list(STATES.keys())[action]

        if new_pos not in SAFE_TRANSITIONS[self.position]:
            reward = REWARD_UNSAFE
        else:
            self.position = new_pos

            if new_pos == 'stirrer' and not self.holding_stirrer and not self.mixed:
                self.holding_stirrer = True
                reward = REWARD_SAISIE
                print(f"  ✓ AGITATEUR SAISI (attach) !")

            elif new_pos == 'mix' and self.holding_stirrer and not self.mixed:
                self.mixed = True
                reward = REWARD_MELANGE
                print(f"  ✓ MÉLANGE EFFECTUÉ (agitateur toujours tenu) !")

            elif new_pos == 'stirrer' and self.holding_stirrer and self.mixed:
                self.holding_stirrer = False
                reward = REWARD_DEPOT
                self.done = True
                info["success"] = True
                print(f"  ✓ AGITATEUR DÉTACHÉ — MISSION TERMINÉE ! 🎉")

        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        return self._get_obs(), reward, self.done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 'home'
        self.holding_stirrer = False
        self.mixed = False
        self.step_count = 0
        self.done = False
        return self._get_obs(), {}

    def render(self):
        print(f"  Pos={self.position} | Tenu={self.holding_stirrer} | Mélangé={self.mixed}")

    def close(self):
        pass


if __name__ == "__main__":
    env = MyBuddyEnvSequenceRight()
    obs, _ = env.reset()
    for action in [1, 2, 3, 4, 3, 2]:
        obs, r, done, trunc, info = env.step(action)
        print(f"Action: {info['action']:15s} | Reward: {r:+6.1f}")
        env.render()
