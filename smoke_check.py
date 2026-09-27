"""Packaging checks only; no training and no reported research metrics."""
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from mybuddy_env_sequence_v3 import MyBuddyEnvSequenceV3
from mybuddy_env_sequence_right import MyBuddyEnvSequenceRight


def check(cls):
    with redirect_stdout(StringIO()):
        env = cls()
        initial, _ = env.reset()
        assert initial.shape == (7,) and env.observation_space.contains(initial)
        observation, reward, done, truncated, info = env.step(4)
        assert reward == -20 and not done and not truncated
        assert np.array_equal(initial, observation), 'Invalid move changed state'
        env.reset()
        rewards = []
        for action in [1, 2, 3, 4, 3, 2]:
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
        assert rewards == [-1, 5, -1, 20, -1, 100]
        assert done and info['success'] and not truncated
        assert observation[-2] == 0 and observation[-1] == 1
        env.reset()
        for _ in range(15):
            _, _, done, truncated, info = env.step(0)
        assert truncated and not done and not info['success']
        env.close()
    print(cls.__name__ + ': success path, illegal move and timeout checks passed')


if __name__ == '__main__':
    for cls in (MyBuddyEnvSequenceV3, MyBuddyEnvSequenceRight):
        check(cls)
