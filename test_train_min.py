# test_train_min.py
import time
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

class TinyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # headless: DIRECT
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(".")

        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.step_cnt = 0
    def reset(self, seed=None, options=None):
        self.step_cnt = 0
        return np.array([0.0], dtype=np.float32), {}
    def step(self, action):
        self.step_cnt += 1
        obs = np.array([random.random()], dtype=np.float32)
        reward = 0.0
        truncated = self.step_cnt > 100
        terminated = False
        return obs, reward, terminated, truncated, {}
    def close(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass

def run():
    vec_env = make_vec_env(lambda: TinyEnv(), n_envs=4, vec_env_cls=DummyVecEnv)
    model = PPO("MlpPolicy", vec_env, verbose=0)
    try:
        model.learn(total_timesteps=2000)
        print("training finished")
    finally:
        vec_env.close()

if __name__ == "__main__":
    run()

