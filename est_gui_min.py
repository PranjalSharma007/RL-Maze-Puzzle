# test_gui_min.py
import time
import pybullet as p
import pybullet_data

def run():
    client = p.connect(p.GUI)
    print("connected client:", client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10, physicsClientId=client)

    # simple floor + sphere
    plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
    plane_id = p.createMultiBody(baseCollisionShapeIndex=plane, physicsClientId=client)
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=client)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=client)
    ball = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_col,
                             baseVisualShapeIndex=sphere_vis, basePosition=[0,0,1], physicsClientId=client)

    try:
        for i in range(10000):
            p.stepSimulation(physicsClientId=client)
            if i % 100 == 0:
                print("step", i)
            time.sleep(1.0/240.0)
    finally:
        p.disconnect(client)
        print("done")

if __name__ == "__main__":
    run()

