"""
mybuddy_env_sequence_v3.py — DQN avec séquence complète et geste de versement
Positions : home, waypoint1, beaker, waypoint2, pre_pour

Logique :
  - 1ère arrivée à 'beaker'    -> ATTACHE (saisie du bécher)
  - arrivée à 'pre_pour'       -> VERSEMENT (gripper ouvre, reste attaché)
  - 2e arrivée à 'beaker'      -> DÉTACHE (fin de mission)
"""
import gymnasium as gym
import numpy as np

STATES = {
    'home'      : 0,
    'waypoint1' : 1,
    'beaker'    : 2,
    'waypoint2' : 3,
    'pre_pour'  : 4,
}
N_POSITIONS = len(STATES)

ACTIONS = {
    0: 'go_home',
    1: 'go_waypoint1',
    2: 'go_beaker',
    3: 'go_waypoint2',
    4: 'go_pre_pour',
}
N_ACTIONS = len(ACTIONS)
N_OBS = N_POSITIONS + 2  # one-hot position + holding_beaker + poured

# Transitions autorisées : uniquement entre positions adjacentes
# (respecte le chemin physique réel du bras, évite les sauts dangereux)
SAFE_TRANSITIONS = {
    'home'      : ['home', 'waypoint1'],
    'waypoint1' : ['waypoint1', 'home', 'beaker'],
    'beaker'    : ['beaker', 'waypoint1', 'waypoint2'],
    'waypoint2' : ['waypoint2', 'beaker', 'pre_pour'],
    'pre_pour'  : ['pre_pour', 'waypoint2'],
}

REWARD_SAISIE       =   5.0
REWARD_VERSEMENT    =  20.0
REWARD_DEPOT        = 100.0
REWARD_STEP_PENALTY =  -1.0
REWARD_UNSAFE       = -20.0
MAX_STEPS           =  15


class MyBuddyEnvSequenceV3(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.position = 'home'
        self.holding_beaker = False
        self.poured = False
        self.step_count = 0
        self.done = False
        print(f"MyBuddyEnvSequenceV3 (5 positions, attach/pour/detach) ✓")

    def _get_obs(self):
        pos_one_hot = np.zeros(N_POSITIONS)
        pos_one_hot[STATES[self.position]] = 1.0
        return np.concatenate([
            pos_one_hot,
            [float(self.holding_beaker)],
            [float(self.poured)],
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

            if new_pos == 'beaker' and not self.holding_beaker and not self.poured:
                # 1ère arrivée : attache
                self.holding_beaker = True
                reward = REWARD_SAISIE
                print(f"  ✓ BÉCHER SAISI (attach) !")

            elif new_pos == 'pre_pour' and self.holding_beaker and not self.poured:
                # Versement : reste attaché
                self.poured = True
                reward = REWARD_VERSEMENT
                print(f"  ✓ VERSEMENT EFFECTUÉ (bécher toujours tenu) !")

            elif new_pos == 'beaker' and self.holding_beaker and self.poured:
                # 2e arrivée après versement : détache, fin
                self.holding_beaker = False
                reward = REWARD_DEPOT
                self.done = True
                info["success"] = True
                print(f"  ✓ BÉCHER DÉTACHÉ — MISSION TERMINÉE ! 🎉")

        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        return self._get_obs(), reward, self.done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 'home'
        self.holding_beaker = False
        self.poured = False
        self.step_count = 0
        self.done = False
        return self._get_obs(), {}

    def render(self):
        print(f"  Pos={self.position} | Tenu={self.holding_beaker} | Versé={self.poured}")

    def close(self):
        pass


if __name__ == "__main__":
    env = MyBuddyEnvSequenceV3()
    obs, _ = env.reset()
    # Séquence idéale : home -> waypoint1 -> beaker -> waypoint2 -> pre_pour -> waypoint2 -> beaker
    for action in [1, 2, 3, 4, 3, 2]:
        obs, r, done, trunc, info = env.step(action)
        print(f"Action: {info['action']:15s} | Reward: {r:+6.1f}")
        env.render()
