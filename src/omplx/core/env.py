from omplx.core.robot import Robot
from omplx.core.obstacle import Obstacle
from omplx.utils.conversion import state_to_jax_array
from typing import List
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Array, Int, Bool
from pyroki.collision._geometry import CollGeom
from typing import Callable

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    import sys
    sys.path.append('./ompl/build/bindings/')
    from ompl import base as ob
    from ompl import geometric as og

@jdc.pytree_dataclass
class JAXEnv: 
    robot: Robot 
    obstacles: CollGeom
    self_collision_margin: jdc.Static[float] = None
    world_collision_margin: jdc.Static[float] = None
    robot_joint_limits_lower: jdc.Static[Float[Array, "n"]] = None
    robot_joint_limits_upper: jdc.Static[Float[Array, "n"]] = None
    self_collision_fn_jit: jdc.Static[Callable[[Float[Array, "n"]], Float[Array, "m"]]] = None
    world_collision_fn_jit: jdc.Static[Callable[[Float[Array, "n"], CollGeom], Float[Array, "m"]]] = None
    n_checks: jdc.Static[Int] = 10

    @classmethod
    def from_robot_and_obstacles(cls, robot: Robot, obstacles: Obstacle, self_collision_margin: float=0.01, world_collision_margin: float=0.01, n_checks: Int=10):
        return cls(
            robot=robot,
            obstacles=obstacles.coll_geom,
            self_collision_margin=self_collision_margin,
            world_collision_margin=world_collision_margin,
            robot_joint_limits_lower=robot.joint_limits_lower,
            robot_joint_limits_upper=robot.joint_limits_upper,
            self_collision_fn_jit=jax.jit(robot.compute_self_collision_distance),
            world_collision_fn_jit=jax.jit(robot.compute_world_collision_distance),
            n_checks=n_checks,
        )
        
    def _check_robot_configuration_validity(self, q: Float[Array, "n"]) -> bool:
        # Check joint limits
        if jnp.any(q < self.robot_joint_limits_lower) or jnp.any(q > self.robot_joint_limits_upper):
            return False
        
        # Check self-collision
        self_distances = self.self_collision_fn_jit(q)
        if jnp.any(self_distances < self.self_collision_margin):
            # print(f"Minimum distance to self-collision: {jnp.min(self_distances)}")
            return False
        
        # Check world collision
        world_distances = self.world_collision_fn_jit(q, self.obstacles)
        if jnp.any(world_distances < self.world_collision_margin):
            # print(f"Minimum distance to world collision: {jnp.min(world_distances)}")
            return False
        
        return True

    @jdc.jit
    def _check_robot_configuration_validity_jit(self, q: Float[Array, "n"]) -> bool:
        """Check if a robot configuration is valid.
        
        Args:
            q: Joint configuration
            
        Returns:
            True if configuration is valid
        """
        limits_violated = jnp.any(q < self.robot_joint_limits_lower) | jnp.any(q > self.robot_joint_limits_upper)
        self_collision_violated = jnp.any(self.self_collision_fn_jit(q) < self.self_collision_margin)
        world_collision_violated = jnp.any(self.world_collision_fn_jit(q, self.obstacles) < self.world_collision_margin)
        return ~limits_violated & ~self_collision_violated & ~world_collision_violated
    
    @jdc.jit
    def _check_motion_validity_linear_jit(self, q1: Float[Array, "n"], q2: Float[Array, "n"]) -> Bool[Array, "n_checks"]:

        q_interp = self.robot.interpolate_joint_configuration_linear(q1, q2, self.n_checks)
        return jax.vmap(self._check_robot_configuration_validity_jit)(q_interp)

class OXMotionValidator(ob.MotionValidator):
    def __init__(self, si, env: JAXEnv):
        super().__init__(si)
        self.env = env

    def checkMotion(self, s1, s2) -> bool:
        q1 = state_to_jax_array(s1, self.env.robot.num_joints)
        q2 = state_to_jax_array(s2, self.env.robot.num_joints)
        return bool(jnp.all(self.env._check_motion_validity_linear_jit(q1, q2)))

def make_state_validity_checker(env: JAXEnv):
    def is_valid(state):
        q = state_to_jax_array(state, env.robot.num_joints)
        return bool(env._check_robot_configuration_validity_jit(q))
    return is_valid