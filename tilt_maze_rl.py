"""
Laptop-only INTERACTIVE 3-D TILT-MAZE  +  ON-THE-FLY RL  (PPO)
Python 3.9+  |  pip install pybullet numpy gymnasium stable-baselines3[pytorch] dearpygui
"""
import pybullet as p
import pybullet_data
import numpy as np
import dearpygui.dearpygui as dpg
import threading, time, math, random, os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

# -------------------------------------------------
# 1.  GYMNASIUM ENV  (tilt angles = continuous action)
# -------------------------------------------------
class TiltMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else None
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # ---- create maze ----
        self.maze_size = 7          # squares
        self.wall_h    = 1.0
        self.goal_pos  = np.array([self.maze_size-1, self.maze_size-1], dtype=float)
        self._build_maze()
        self.ball = self._spawn_ball()

        # spaces
        obs_low  = np.array([0, 0, -20, -20], dtype=np.float32)
        obs_high = np.array([self.maze_size, self.maze_size, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # pitch, roll (deg)

        self.step_cnt = 0
        self.max_episode_steps = 400

    # ---------- maze construction ----------
    def _build_maze(self):
        # simple grid with random holes
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, self.maze_size/2, 0.05])
        p.createMultiBody(baseCollisionShapeIndex=plane, basePosition=[self.maze_size/2, self.maze_size/2, 0])

        for x in range(self.maze_size+1):
            for y in range(self.maze_size+1):
                if random.random() < 0.25 and not (x == 0 and y == 0) and not (x == self.maze_size and y == self.maze_size):
                    continue   # hole
                # vertical wall
                wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.5, self.wall_h/2])
                p.createMultiBody(baseCollisionShapeIndex=wall,
                                  basePosition=[x, y+0.5, self.wall_h/2])
                # horizontal wall
                wall2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.1, self.wall_h/2])
                p.createMultiBody(baseCollisionShapeIndex=wall2,
                                  basePosition=[x+0.5, y, self.wall_h/2])
        # goal cylinder
        goal = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.1)
        p.createMultiBody(baseCollisionShapeIndex=goal,
                          basePosition=[*self.goal_pos, 0.1])

    def _spawn_ball(self):
        ball = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1,0,0,1]),
            basePosition=[0.5, 0.5, 1])
        return ball

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.ball, [0.5, 0.5, 1], [0,0,0,1])
        self.step_cnt = 0
        return self._get_obs(), {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball)
        return np.array([pos[0], pos[1], 0, 0], dtype=np.float32)

    def step(self, action):
        # action is in [-1,1]  ->  map to [-20 deg, 20 deg]
        pitch_deg = np.clip(action[0]*20, -20, 20)
        roll_deg  = np.clip(action[1]*20, -20, 20)
        gx = 9.8 * math.sin(math.radians(roll_deg))
        gy = 9.8 * math.sin(math.radians(pitch_deg))
        p.setGravity(gx, gy, -9.8)

        p.stepSimulation()
        self.step_cnt += 1

        pos, _ = p.getBasePositionAndOrientation(self.ball)
        dist = np.linalg.norm(np.array(pos[:2]) - self.goal_pos)
        reward = -1.0 - 0.2*dist
        terminated = bool(dist < 0.3)
        truncated  = self.step_cnt >= self.max_episode_steps
        obs = np.array([pos[0], pos[1], pitch_deg, roll_deg], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass   # we use the shared GUI window

    def close(self):
        p.disconnect()

# -------------------------------------------------
# 2.  SHARED GLOBALS  (GUI + RL thread)
# -------------------------------------------------
env_render = None          # will hold the env used for visuals
policy     = None          # PPO model
stop_train = threading.Event()

# -------------------------------------------------
# 3.  DEARPYGUI  INTERFACE
# -------------------------------------------------
dpg.create_context()
dpg.create_viewport(title="3-D Tilt-Maze  RL", width=1300, height=850)

with dpg.window(tag="main_win", label="Controls"):
    dpg.add_text("WASD / Arrows  =  manual tilt        (Hold Shift for fine)")
    dpg.add_checkbox(label="Train RL (PPO) in background", default_value=True, tag="train_cb")
    dpg.add_button(label="Reset Ball", callback=lambda: env_render.reset())
    dpg.add_slider_float(label="Pitch (deg)", default_value=0, min_value=-20, max_value=20,
                         tag="pitch_s", callback=lambda: update_gravity())
    dpg.add_slider_float(label="Roll  (deg)", default_value=0, min_value=-20, max_value=20,
                         tag="roll_s",  callback=lambda: update_gravity())
    dpg.add_text("Episode reward: --", tag="rew_txt")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_win", True)

def update_gravity():
    pitch = math.radians(dpg.get_value("pitch_s"))
    roll  = math.radians(dpg.get_value("roll_s"))
    gx = 9.8 * math.sin(roll)
    gy = 9.8 * math.sin(pitch)
    p.setGravity(gx, gy, -9.8)

def key_handler(sender, app_data):
    fine = 0.2 if dpg.is_key_down(dpg.mvKey_Shift) else 1.0
    if app_data == dpg.mvKey_W or app_data == dpg.mvKey_Up:
        dpg.set_value("pitch_s", dpg.get_value("pitch_s") + fine)
    elif app_data == dpg.mvKey_S or app_data == dpg.mvKey_Down:
        dpg.set_value("pitch_s", dpg.get_value("pitch_s") - fine)
    elif app_data == dpg.mvKey_A or app_data == dpg.mvKey_Left:
        dpg.set_value("roll_s",  dpg.get_value("roll_s") - fine)
    elif app_data == dpg.mvKey_D or app_data == dpg.mvKey_Right:
        dpg.set_value("roll_s",  dpg.get_value("roll_s") + fine)
    update_gravity()

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=key_handler)

# -------------------------------------------------
# 4.  BACKGROUND TRAINING THREAD
# -------------------------------------------------
def train_loop():
    global policy
    vec_env = make_vec_env(lambda: TiltMazeEnv(render=False), n_envs=4)
    policy = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tb")
    while not stop_train.is_set():
        policy.learn(total_timesteps=5000)   # small chunks
        policy.save("ppo_tilt_maze")
        print("saved ppo_tilt_maze.zip")

# -------------------------------------------------
# 5.  MAIN RENDER / INTERACTION LOOP
# -------------------------------------------------
env_render = TiltMazeEnv(render=True)
env_render.reset()
train_thread = threading.Thread(target=train_loop, daemon=True)
train_thread.start()

ep_reward = 0.0
obs = env_render._get_obs()

while dpg.is_dearpygui_running():
    # decide control source
    if dpg.get_value("train_cb") and policy is not None:
        # RL drives
        action, _ = policy.predict(obs, deterministic=False)
        pitch_deg = np.clip(action[0]*20, -20, 20)
        roll_deg  = np.clip(action[1]*20, -20, 20)
        dpg.set_value("pitch_s", pitch_deg)
        dpg.set_value("roll_s",  roll_deg)
    else:
        # manual
        pitch_deg = dpg.get_value("pitch_s")
        roll_deg  = dpg.get_value("roll_s")

    # step physics
    gx = 9.8 * math.sin(math.radians(roll_deg))
    gy = 9.8 * math.sin(math.radians(pitch_deg))
    p.setGravity(gx, gy, -9.8)
    p.stepSimulation()

    obs = env_render._get_obs()
    dist = np.linalg.norm(obs[:2] - env_render.goal_pos)
    reward = -1.0 - 0.2*dist + (100.0 if dist < 0.3 else 0.0)
    ep_reward += reward

    # auto-reset
    if dist < 0.3 or env_render.step_cnt >= env_render.max_episode_steps:
        env_render.reset()
        dpg.set_value("rew_txt", f"Last episode reward: {ep_reward:.1f}")
        ep_reward = 0.0

    time.sleep(1/240.)   # PyBullet default
    dpg.render()

# clean-up
stop_train.set()
train_thread.join(timeout=1)
env_render.close()
dpg.destroy_context()
