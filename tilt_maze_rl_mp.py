"""
Interactive Tilt-Maze GUI
- Manual tilt using DearPyGui sliders
- Toggle RL agent control
- Ball visible (bright, bigger radius)
"""

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os, time, math
import numpy as np
import pybullet as p
import pybullet_data
import dearpygui.dearpygui as dpg

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = "ppo_tilt_maze.zip"


# ---------------------------
#  TiltMazeEnv
# ---------------------------
class TiltMazeEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10, physicsClientId=self.client)

        self.maze_size = 7
        self.wall_h = 1.0
        self.goal_pos = np.array([6.5, 6.5], dtype=float)
        self.start_pos = np.array([0.5, 0.5], dtype=float)

        self._build_maze()
        self.ball = self._spawn_ball()

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -20, -20], dtype=np.float32),
            high=np.array([self.maze_size, self.maze_size, 20, 20], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def _build_maze(self):
        # Base floor
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, self.maze_size/2, 0.05])
        p.createMultiBody(baseCollisionShapeIndex=plane,
                          basePosition=[self.maze_size/2, self.maze_size/2, 0])

        # Goal marker
        goal_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.05, rgbaColor=[0,1,0,0.6])
        p.createMultiBody(baseVisualShapeIndex=goal_vis,
                          basePosition=[*self.goal_pos, 0.05])

    def _spawn_ball(self):
        coll = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)  # bigger ball
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0.9, 0.1, 0.1, 1])
        ball = p.createMultiBody(baseMass=0.1,
                                 baseCollisionShapeIndex=coll,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[*self.start_pos, 0.5])
        return ball

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.ball, [*self.start_pos, 0.5], [0,0,0,1])
        p.resetBaseVelocity(self.ball, [0,0,0], [0,0,0])
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball)
        vel, _ = p.getBaseVelocity(self.ball)
        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)

    def step(self, action):
        pitch_deg = np.clip(action[0] * 15, -15, 15)
        roll_deg  = np.clip(action[1] * 15, -15, 15)
        gx = -9.8 * math.sin(math.radians(roll_deg))
        gy =  9.8 * math.sin(math.radians(pitch_deg))
        p.setGravity(gx, gy, -9.8)

        for _ in range(4):
            p.stepSimulation()

        return self._get_obs(), 0.0, False, False, {}

    def close(self):
        p.disconnect(self.client)


# ---------------------------
# GUI & Main
# ---------------------------
def main():
    global env
    env = TiltMazeEnv(render=True)
    obs, _ = env.reset()

    dpg.create_context()
    dpg.create_viewport(title="Interactive Tilt-Maze", width=400, height=300)

    with dpg.window(tag="main_win"):
        dpg.add_text("Tilt Controls")
        dpg.add_slider_float(label="Pitch", tag="pitch", default_value=0.0, min_value=-1.0, max_value=1.0)
        dpg.add_slider_float(label="Roll", tag="roll", default_value=0.0, min_value=-1.0, max_value=1.0)
        dpg.add_checkbox(label="Use RL Agent", tag="rl_cb")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_win", True)

    policy = PPO.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    while dpg.is_dearpygui_running():
        if dpg.get_value("rl_cb") and policy:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.array([dpg.get_value("pitch"), dpg.get_value("roll")])

        obs, _, _, _, _ = env.step(action)
        dpg.render_dearpygui_frame()
        time.sleep(1.0/60)

    env.close()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

