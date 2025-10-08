"""
Autonomous Robot Maze Project
-> Fixed maze with walls & boundaries
-> Ball (robot) as agent
-> Drone-like top-down camera
-> GUI with Reset, Manual tilt, RL toggle
"""

import os, time, math
import numpy as np
import pybullet as p
import pybullet_data
import dearpygui.dearpygui as dpg

from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym


MODEL_PATH = "ppo_tilt_maze.zip"


# --------------------------
#  TiltMazeEnvironment
# --------------------------
class TiltMazeEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        self.maze_size = 15
        self.wall_h = 2.0
        self.goal_pos = np.array([13.5, 13.5], dtype=float)
        self.start_pos = np.array([1.5, 1.5], dtype=float)

        self._build_maze()
        self.ball = self._spawn_ball()

        # Camera = drone view
        p.resetDebugVisualizerCamera(cameraDistance=25,
                                     cameraYaw=90,
                                     cameraPitch=-89,
                                     cameraTargetPosition=[self.maze_size/2, self.maze_size/2, 0],
                                     physicsClientId=self.client)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, -20, -20], dtype=np.float32),
            high=np.array([self.maze_size, self.maze_size, 20, 20], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def _build_maze(self):
        # Base floor
        plane = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, self.maze_size/2, 0.1], physicsClientId=self.client)
        p.createMultiBody(baseCollisionShapeIndex=plane,
                          basePosition=[self.maze_size/2, self.maze_size/2, -0.1],
                          physicsClientId=self.client)

        # Outer boundary walls
        thickness = 0.5
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.maze_size/2, thickness, self.wall_h/2], physicsClientId=self.client)
        # bottom
        p.createMultiBody(baseCollisionShapeIndex=wall_shape,
                          basePosition=[self.maze_size/2, 0, self.wall_h/2],
                          physicsClientId=self.client)
        # top
        p.createMultiBody(baseCollisionShapeIndex=wall_shape,
                          basePosition=[self.maze_size/2, self.maze_size, self.wall_h/2],
                          physicsClientId=self.client)
        # left
        wall_shape2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, self.maze_size/2, self.wall_h/2], physicsClientId=self.client)
        p.createMultiBody(baseCollisionShapeIndex=wall_shape2,
                          basePosition=[0, self.maze_size/2, self.wall_h/2],
                          physicsClientId=self.client)
        # right
        p.createMultiBody(baseCollisionShapeIndex=wall_shape2,
                          basePosition=[self.maze_size, self.maze_size/2, self.wall_h/2],
                          physicsClientId=self.client)

        # Inner maze walls (fixed layout, simple corridors)
        def add_wall(x, y, dx, dy):
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dx/2, dy/2, self.wall_h/2], physicsClientId=self.client)
            p.createMultiBody(baseCollisionShapeIndex=shape,
                              basePosition=[x + dx/2, y + dy/2, self.wall_h/2],
                              physicsClientId=self.client)

        # Example: add corridors (you can expand with more for a real maze)
        add_wall(3, 0, 0.5, 10)   # vertical
        add_wall(7, 5, 0.5, 10)   # vertical
        add_wall(0, 7, 10, 0.5)   # horizontal
        add_wall(5, 12, 10, 0.5)  # horizontal

        # Goal marker
        goal_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=0.05, rgbaColor=[0,1,0,0.6], physicsClientId=self.client)
        p.createMultiBody(baseVisualShapeIndex=goal_vis,
                          basePosition=[*self.goal_pos, 0.05],
                          physicsClientId=self.client)

    def _spawn_ball(self):
        coll = p.createCollisionShape(p.GEOM_SPHERE, radius=0.4, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=self.client)
        return p.createMultiBody(baseMass=0.2,
                                 baseCollisionShapeIndex=coll,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[*self.start_pos, 0.5],
                                 physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(self.ball, [*self.start_pos, 0.5], [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball, [0,0,0], [0,0,0], physicsClientId=self.client)
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

        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

        return self._get_obs(), 0.0, False, False, {}

    def close(self):
        p.disconnect(self.client)


# -------------------------
# GUI & Main
# ------------------------
def main():
    env = TiltMazeEnv(render=True)
    obs, _ = env.reset()

    dpg.create_context()
    dpg.create_viewport(title="Maze Robot Control", width=400, height=300)

    with dpg.window(tag="main_win"):
        dpg.add_text("Maze Control Panel")
        dpg.add_slider_float(label="Pitch", tag="pitch", default_value=0.0, min_value=-1.0, max_value=1.0)
        dpg.add_slider_float(label="Roll", tag="roll", default_value=0.0, min_value=-1.0, max_value=1.0)
        dpg.add_checkbox(label="Use RL Agent", tag="rl_cb")
        dpg.add_button(label="Reset Robot", callback=lambda: env.reset())

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

