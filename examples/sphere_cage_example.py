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

    # Visualization
    server = viser.ViserServer()
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    
    # Add a key directional light (main light from above-front)
    server.scene.add_light_directional(
        name="/lights/key_light",
        color=(255, 255, 245),  # Warm white
        intensity=2.5,
        cast_shadow=True,
        position=(2.0, 2.0, 3.0),
        wxyz=(0.924, -0.383, 0.0, 0.0),  # Pointing downward-ish
    )
    
    # Add a fill light (softer, from the side, no shadow)
    server.scene.add_light_directional(
        name="/lights/fill_light",
        color=(200, 220, 255),  # Cool white/blue tint
        intensity=1.0,
        cast_shadow=False,  # No shadow for fill light
        position=(-2.0, 1.0, 2.0),
        wxyz=(0.924, 0.383, 0.0, 0.0),
    )
    
    # Add ambient light for overall illumination
    server.scene.add_light_ambient(
        name="/lights/ambient",
        color=(255, 255, 255),
        intensity=0.4,  # Subtle ambient
    )
    
    # Add hemisphere light for better depth perception
    server.scene.add_light_hemisphere(
        name="/lights/hemisphere",
        sky_color=(135, 206, 235),  # Sky blue
        ground_color=(139, 90, 43),  # Brown ground
        intensity=0.5,
    )
    
    # Add a point light near the workspace for highlights
    server.scene.add_light_point(
        name="/lights/point",
        color=(255, 230, 200),
        intensity=1.5,
        distance=5.0,
        cast_shadow=True,
        position=(0.5, 0.5, 1.5),
    )
    obstacle_mesh = obstacles.coll_geom.to_trimesh()
    server.scene.add_mesh_simple(
        name="/obstacle/mesh",
        vertices=obstacle_mesh.vertices,
        faces=obstacle_mesh.faces,
        color=(255, 100, 100),  # Red obstacles
        wireframe=False,
        opacity=0.9,
        cast_shadow=True,
        receive_shadow=True,
    )
    viser_urdf = ViserUrdf(server, robot.urdf, root_node_name="/base")

    while True: 
        for cfg in jax_path:
            viser_urdf.update_cfg(onp.array(cfg))
            time.sleep(0.1)
        time.sleep(1.0)
if __name__ == "__main__":
    main()