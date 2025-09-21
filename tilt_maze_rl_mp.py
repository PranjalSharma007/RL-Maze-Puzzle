#!/usr/bin/env python3
"""
Tilt-maze RL with PyBullet GUI isolated from training process.
Training runs in a separate process to avoid native/Metal/PyBullet races.
"""

import multiprocessing
# Use spawn to be safe on macOS
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import time
import math
import random
import threading
import numpy as np

# PyBullet and rendering + UI imports (these will be used only in main process)
import pybullet as p
import pybullet_data
import dearpygui.dearpygui as dpg

# Training imports (also used in child process)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = "ppo_tilt_maze.zip"


# ---------------------------
#  TiltMazeEnv (safe, uses physicsClientId everywhere)
# ---------------------------
class TiltMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        # each env keeps its own client id
        try:
            self.client = p.connect(p.GUI if render else p.DIRECT)
        except Exception:
            # fallback
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10, physicsClientId=self.client)

        self.maze_size = 7
        self.wall_h = 1.0
        self.goal_pos = np.array([self.maze_size - 1, self.maze_size - 1], dtype=float)
        self.maze_body_ids = self._build_maze()
        self.ball = self._spawn_ball()

        obs_low = np.array([0, 0, -20, -20], dtype=np.float32)
        obs_high = np.array([self.maze_size, self.maze_size, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.step_cnt = 0
        self.max_episode_steps = 400

    def _build_maze(self):
        body_ids = []
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, self.maze_size/2, 0.05],
                                       physicsClientId=self.client)
        plane_id = p.createMultiBody(baseCollisionShapeIndex=plane, basePosition=[self.maze_size/2, self.maze_size/2, 0],
                                     physicsClientId=self.client)
        body_ids.append(plane_id)

        wall_positions = [
            ([self.maze_size/2, 0, self.wall_h/2], [self.maze_size/2, 0.1, self.wall_h/2]),
            ([self.maze_size/2, self.maze_size, self.wall_h/2], [self.maze_size/2, 0.1, self.wall_h/2]),
            ([0, self.maze_size/2, self.wall_h/2], [0.1, self.maze_size/2, self.wall_h/2]),
            ([self.maze_size, self.maze_size/2, self.wall_h/2], [0.1, self.maze_size/2, self.wall_h/2])
        ]
        for pos, ext in wall_positions:
            w = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext, physicsClientId=self.client)
            wid = p.createMultiBody(baseCollisionShapeIndex=w, basePosition=pos, physicsClientId=self.client)
            body_ids.append(wid)

        for x in range(1, self.maze_size):
            for y in range(1, self.maze_size):
                if random.random() < 0.35:
                    if random.random() > 0.5:
                        wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.5, self.wall_h/2],
                                                      physicsClientId=self.client)
                        wall_id = p.createMultiBody(baseCollisionShapeIndex=wall,
                                                    basePosition=[x, y+0.5, self.wall_h/2],
                                                    physicsClientId=self.client)
                    else:
                        wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, self.wall_h/2],
                                                      physicsClientId=self.client)
                        wall_id = p.createMultiBody(baseCollisionShapeIndex=wall,
                                                    basePosition=[x+0.5, y, self.wall_h/2],
                                                    physicsClientId=self.client)
                    body_ids.append(wall_id)

        goal_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.02, rgbaColor=[0,1,0,0.5],
                                          physicsClientId=self.client)
        p.createMultiBody(baseVisualShapeIndex=goal_visual, basePosition=[*self.goal_pos, 0.06],
                          physicsClientId=self.client)
        return body_ids

    def _spawn_ball(self):
        coll = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.8, 0.1, 0.1, 1], physicsClientId=self.client)
        ball = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=coll, baseVisualShapeIndex=vis,
                                 basePosition=[0.5, 0.5, 0.5], physicsClientId=self.client)
        p.changeDynamics(ball, -1, spinningFriction=0.001, rollingFriction=0.001, linearDamping=0.0,
                         physicsClientId=self.client)
        return ball

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.ball, [0.5, 0.5, 0.5], [0,0,0,1], physicsClientId=self.client)
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
        p.stepSimulation(physicsClientId=self.client)
        self.step_cnt += 1

        obs = self._get_obs()
        pos = obs[:2]
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)
        reward = -0.1 * dist_to_goal
        terminated = bool(dist_to_goal < 0.3)
        if terminated:
            reward += 500.0
        truncated = self.step_cnt >= self.max_episode_steps
        if truncated:
            reward -= 10.0
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        try:
            p.disconnect(self.client)
        except Exception:
            pass


# ---------------------------
# Training process target (runs in separate process)
# ---------------------------
def training_process_fn(stop_event, training_active_event, model_path):
    """
    This function runs in a separate process.
    It creates its own DIRECT PyBullet envs and trains a PPO model.
    """
    print("[TRAIN PROC] starting")
    # create vectorized envs in this process
    vec_env = make_vec_env(lambda: TiltMazeEnv(render=False), n_envs=4, vec_env_cls=DummyVecEnv)

    # load or create model
    if os.path.exists(model_path):
        print("[TRAIN PROC] loading existing model")
        train_policy = PPO.load(model_path, env=vec_env)
    else:
        print("[TRAIN PROC] creating new model")
        train_policy = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./tilt_maze_tensorboard/")

    try:
        while not stop_event.is_set():
            if training_active_event.is_set():
                # perform shorter learns and save often so main process can pick up models
                train_policy.learn(total_timesteps=5000, reset_num_timesteps=False)
                train_policy.save(model_path)
                print("[TRAIN PROC] saved model")
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            train_policy.save(model_path)
        except Exception:
            pass
        print("[TRAIN PROC] exiting")


# ---------------------------
# Main GUI process (this runs by default)
# ---------------------------
def main():
    # multiproc events (works across spawn)
    stop_event = multiprocessing.Event()
    training_active_event = multiprocessing.Event()

    # Start training process
    train_proc = multiprocessing.Process(
        target=training_process_fn,
        args=(stop_event, training_active_event, MODEL_PATH),
        daemon=True
    )
    train_proc.start()
    print("[MAIN] started training process:", train_proc.pid)

    # Setup DearPyGui
    dpg.create_context()
    dpg.create_viewport(title="3-D Tilt-Maze RL (main)", width=900, height=650)

    # GUI elements
    def toggle_training_cb(sender, app_data):
        if app_data:
            training_active_event.set()
        else:
            training_active_event.clear()

    with dpg.window(tag="main_win"):
        dpg.add_text("Manual Control: WASD / Arrow Keys (Hold Shift for fine-tuning)")
        dpg.add_checkbox(label="Enable RL Agent Control", tag="train_cb", callback=toggle_training_cb)
        dpg.add_button(label="Reset Ball", callback=lambda: env_render.reset())
        dpg.add_slider_float(label="Pitch (W/S)", tag="pitch_s", min_value=-15, max_value=15, callback=lambda s,a,u: None)
        dpg.add_slider_float(label="Roll (A/D)", tag="roll_s", min_value=-15, max_value=15, callback=lambda s,a,u: None)
        dpg.add_text("Last Episode Reward: --", tag="rew_txt")
        dpg.add_text("Total Timesteps: 0", tag="ts_txt")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_win", True)

    # Key handler for sliders
    def key_handler(sender, app_data):
        if not dpg.get_value("train_cb"):
            increment = 0.5 if dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift) else 2.0
            if app_data == dpg.mvKey_W or app_data == dpg.mvKey_Up:
                dpg.set_value("pitch_s", np.clip(dpg.get_value("pitch_s") + increment, -15, 15))
            elif app_data == dpg.mvKey_S or app_data == dpg.mvKey_Down:
                dpg.set_value("pitch_s", np.clip(dpg.get_value("pitch_s") - increment, -15, 15))
            elif app_data == dpg.mvKey_A or app_data == dpg.mvKey_Left:
                dpg.set_value("roll_s", np.clip(dpg.get_value("roll_s") - increment, -15, 15))
            elif app_data == dpg.mvKey_D or app_data == dpg.mvKey_Right:
                dpg.set_value("roll_s", np.clip(dpg.get_value("roll_s") + increment, -15, 15))

    with dpg.handler_registry():
        dpg.add_key_down_handler(callback=key_handler)

    # Create GUI-enabled env in main process (only this env uses p.GUI)
    global env_render, policy_render, latest_total_timesteps
    env_render = TiltMazeEnv(render=True)
    obs, _ = env_render.reset()
    last_obs, _ = env_render.reset()
    policy_render = None
    latest_total_timesteps = 0

    # debug: last modification time for model file; used to pick up updated models from train proc
    last_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else -1

    # set camera
    try:
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[3.5, 3.5, 0],
                                    physicsClientId=env_render.client)
    except Exception:
        pass

    try:
        while dpg.is_dearpygui_running():
            # check for updated model by modification time
            if os.path.exists(MODEL_PATH):
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime != last_mtime:
                    last_mtime = mtime
                    # load model for inference only (no env passed)
                    try:
                        policy_render = PPO.load(MODEL_PATH, env=None)
                        print("[MAIN] loaded updated policy")
                    except Exception as e:
                        print("[MAIN] policy load error:", e)

            action = np.array([0.0, 0.0])
            if dpg.get_value("train_cb") and policy_render is not None:
                # try model prediction
                try:
                    action, _ = policy_render.predict(last_obs, deterministic=True)
                    dpg.set_value("pitch_s", np.clip(action[0] * 15, -15, 15))
                    dpg.set_value("roll_s",  np.clip(action[1] * 15, -15, 15))
                except Exception as e:
                    # fallback to manual on error
                    action = np.array([dpg.get_value("pitch_s") / 15.0, dpg.get_value("roll_s") / 15.0])
            else:
                action = np.array([dpg.get_value("pitch_s") / 15.0, dpg.get_value("roll_s") / 15.0])

            obs, reward, terminated, truncated, info = env_render.step(action)
            latest_total_timesteps = latest_total_timesteps  # updated by reading model file timestamps; optional
            last_obs = obs

            # update GUI text
            dpg.set_value("ts_txt", f"Total Timesteps: {latest_total_timesteps}")
            dpg.render_dearpygui_frame()

            if terminated or truncated:
                dpg.set_value("rew_txt", f"Last Episode Reward: {reward:.2f}")
                last_obs, _ = env_render.reset()

            # small sleep to not hog CPU
            time.sleep(1.0 / 120.0)
    finally:
        print("[MAIN] shutting down")
        # signal training process to stop
        stop_event.set()
        # give it a moment to exit
        train_proc.join(timeout=5)
        if train_proc.is_alive():
            print("[MAIN] terminating training process")
            train_proc.terminate()
            train_proc.join(timeout=2)
        try:
            env_render.close()
        except Exception:
            pass
        dpg.destroy_context()


if __name__ == "__main__":
    main()

