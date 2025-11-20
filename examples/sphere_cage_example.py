from omplx.core.env import JAXEnv
from omplx.core.robot import Robot
from omplx.core.obstacle import Obstacle
from omplx.core.planner import MotionPlanner
import jax
import jax.numpy as jnp

# For visualization purposes
import viser
from viser.extras import ViserUrdf
import numpy as onp
import time

robot_urdf_path="resources/panda/panda_spherized.urdf"
robot_mesh_dir="resources/panda/meshes"

# Start, goal, and problem specification from https://github.com/KavrakiLab/vamp/blob/main/scripts/sphere_cage_example.py
# Starting configuration
a = jnp.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785])
# Goal configuration
b = jnp.array([2.35, 1., 0., -0.8, 0, 2.5, 0.785])
# Problem specification: a list of sphere centers
problem = [
    [0.55, 0, 0.25],
    [0.35, 0.35, 0.25],
    [0, 0.55, 0.25],
    [-0.55, 0, 0.25],
    [-0.35, -0.35, 0.25],
    [0, -0.55, 0.25],
    [0.35, -0.35, 0.25],
    [0.35, 0.35, 0.8],
    [0, 0.55, 0.8],
    [-0.35, 0.35, 0.8],
    [-0.55, 0, 0.8],
    [-0.35, -0.35, 0.8],
    [0, -0.55, 0.8],
    [0.35, -0.35, 0.8],
]
radius = 0.2
radii = [radius] * len(problem)

def main():
    robot = Robot.from_urdf(robot_urdf_path, mesh_dir=robot_mesh_dir)
    obstacles = Obstacle.from_sphere(problem, radii)
    env = JAXEnv.from_robot_and_obstacles(robot, obstacles, self_collision_margin=-0.1, world_collision_margin=-0.01, n_checks=10)
    planner = MotionPlanner(env, planner='rrtc')
    path = planner.plan(start=a, goal=b)
    jax_path = planner.path_to_jax_array(path)
    # print(f"Path: {jax_path}")

    # Visualization
    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=obstacles.coll_geom.to_trimesh())
    viser_urdf = ViserUrdf(server, robot.urdf, root_node_name="/base")

    while True: 
        for cfg in jax_path:
            viser_urdf.update_cfg(onp.array(cfg))
            time.sleep(0.1)
if __name__ == "__main__":
    main()