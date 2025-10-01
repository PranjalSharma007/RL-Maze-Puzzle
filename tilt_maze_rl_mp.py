"""
Tilt-Maze RL Environment with PPO Training + Manual/Agent Play
- Fixed maze layout (walls, start, goal)
- Reward shaping: -0.05 per step, -5 for collision, +1 per step closer to goal, +500 at goal
- Continuous control: pitch/roll angles (±15°)
- Faster ball dynamics: multiple physics steps per env step
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
#  TiltMazeEnv (Fixed Maze)
# ---------------------------
class TiltMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10, physicsClientId=self.client)

        self.maze_size = 7
        self.wall_h = 1.0
        self.goal_pos = np.array([6.5, 6.5], dtype=float)  # top-right corner
        self.start_pos = np.array([0.5, 0.5], dtype=float)

        self._build_maze()
        self.ball = self._spawn_ball()

        # obs = (x, y, vx, vy)
        obs_low = np.array([0, 0, -20, -20], dtype=np.float32)
        obs_high = np.array([self.maze_size, self.maze_size, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.step_cnt = 0
        self.max_episode_steps = 400

    def _build_maze(self):
        # Base floor
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, self.maze_size/2, 0.05],
                                       physicsClientId=self.client)
        p.createMultiBody(baseCollisionShapeIndex=plane,
                          basePosition=[self.maze_size/2, self.maze_size/2, 0],
                          physicsClientId=self.client)

        # Outer boundaries
        walls = [
            ([self.maze_size/2, 0, self.wall_h/2], [self.maze_size/2, 0.1, self.wall_h/2]),
            ([self.maze_size/2, self.maze_size, self.wall_h/2], [self.maze_size/2, 0.1, self.wall_h/2]),
            ([0, self.maze_size/2, self.wall_h/2], [0.1, self.maze_size/2, self.wall_h/2]),
            ([self.maze_size, self.maze_size/2, self.wall_h/2], [0.1, self.maze_size/2, self.wall_h/2]),
        ]
        for pos, ext in walls:
            w = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext, physicsClientId=self.client)
            p.createMultiBody(baseCollisionShapeIndex=w, basePosition=pos, physicsClientId=self.client)

        # Internal fixed walls (design a simple maze)
        wall_positions = [
            ([2, 1.5, self.wall_h/2], [2, 0.1, self.wall_h/2]),
            ([4, 3.5, self.wall_h/2], [0.1, 2, self.wall_h/2]),
            ([1.5, 5, self.wall_h/2], [1.5, 0.1, self.wall_h/2]),
            ([5.5, 2.5, self.wall_h/2], [0.1, 2, self.wall_h/2])
        ]
        for pos, ext in wall_positions:
            wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext, physicsClientId=self.client)
            p.createMultiBody(baseCollisionShapeIndex=wall, basePosition=pos, physicsClientId=self.client)

        # Goal marker
        goal_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.02, rgbaColor=[0,1,0,0.6],
                                       physicsClientId=self.client)
        p.createMultiBody(baseVisualShapeIndex=goal_vis, basePosition=[*self.goal_pos, 0.05],
                          physicsClientId=self.client)

    def _spawn_ball(self):
        coll = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=self.client)
        ball = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                 basePosition=[*self.start_pos, 0.5], physicsClientId=self.client)
        return ball

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.ball, [*self.start_pos, 0.5], [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball, [0,0,0], [0,0,0], physicsClientId=self.client)
        self.step_cnt = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.ball, physicsClientId=self.client)
        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)

    def step(self, action):
        pitch_deg = np.clip(action[0] * 15, -15, 15)
        roll_deg  = np.clip(action[1] * 15, -15, 15)
        gx = -9.8 * math.sin(math.radians(roll_deg))
        gy =  9.8 * math.sin(math.radians(pitch_deg))
        p.setGravity(gx, gy, -9.8, physicsClientId=self.client)

        # Step simulation faster
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

        self.step_cnt += 1
        obs = self._get_obs()
        pos = obs[:2]
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)

        # Reward shaping
        reward = -0.05  # step penalty
        reward += (7 - dist_to_goal) * 0.01  # encourage getting closer

        # Detect collision with boundaries
        if pos[0] <= 0.2 or pos[0] >= self.maze_size-0.2 or pos[1] <= 0.2 or pos[1] >= self.maze_size-0.2:
            reward -= 5.0

        terminated = bool(dist_to_goal < 0.3)
        if terminated:
            reward += 500.0
        truncated = self.step_cnt >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(self.client)


# ---------------------------
# Training Process
# ---------------------------
def training_process_fn(stop_event, training_active_event, model_path):
    vec_env = make_vec_env(lambda: TiltMazeEnv(render=False), n_envs=4, vec_env_cls=DummyVecEnv)
    if os.path.exists(model_path):
        policy = PPO.load(model_path, env=vec_env)
    else:
        policy = PPO("MlpPolicy", vec_env, verbose=1)

    while not stop_event.is_set():
        if training_active_event.is_set():
            policy.learn(total_timesteps=5000, reset_num_timesteps=False)
            policy.save(model_path)
        else:
            time.sleep(0.5)
    vec_env.close()
    policy.save(model_path)


# ---------------------------
# GUI & Main
# ---------------------------
def main():
    stop_event = multiprocessing.Event()
    training_active_event = multiprocessing.Event()

    train_proc = multiprocessing.Process(
        target=training_process_fn,
        args=(stop_event, training_active_event, MODEL_PATH),
        daemon=True
    )
    train_proc.start()

    dpg.create_context()
    dpg.create_viewport(title="Tilt-Maze RL", width=800, height=600)
    with dpg.window(tag="main_win"):
        dpg.add_checkbox(label="Enable RL Agent", tag="train_cb",
                         callback=lambda s,a: training_active_event.set() if a else training_active_event.clear())
        dpg.add_button(label="Reset", callback=lambda: env.reset())
        dpg.add_text("Last Reward: --", tag="rew_txt")
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_win", True)

    global env
    env = TiltMazeEnv(render=True)
    obs, _ = env.reset()
    policy = None
    last_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else -1

    while dpg.is_dearpygui_running():
        if os.path.exists(MODEL_PATH):
            mtime = os.path.getmtime(MODEL_PATH)
            if mtime != last_mtime:
                last_mtime = mtime
                policy = PPO.load(MODEL_PATH, env=None)

        action = np.array([0.0, 0.0])
        if dpg.get_value("train_cb") and policy:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        dpg.set_value("rew_txt", f"Reward: {reward:.2f}")
        if term or trunc:
            obs, _ = env.reset()
        dpg.render_dearpygui_frame()
        time.sleep(1.0/120)

    stop_event.set()
    train_proc.join()
    env.close()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

