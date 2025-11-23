"""Robot model wrapper that unifies PyRoki with collision checking.

Provides a clean interface to robot kinematics, dynamics, and collision geometry.
"""

from typing import Optional

import jax.numpy as jnp
from jaxlie import SE3
import pyroki as pk
import yourdfpy
from jax import Array
from jaxtyping import Float, Int
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class Robot:
    # def __init__(
    #     self,
    #     pyroki_robot: pk.Robot,
    #     collision_model: pk.collision.RobotCollision,
    #     urdf: yourdfpy.URDF,
    #     end_effector_link: Optional[str] = None,
    # ):
    #     """Initialize robot model.

    #     Args:
    #         pyroki_robot: PyRoki Robot instance
    #         collision_model: PyRoki collision model
    #         urdf: Original URDF object
    #         end_effector_link: Name of end-effector link (auto-detected if None)
    #     """
    #     self.pyroki_robot: pk.Robot = pyroki_robot
    #     self.collision_model: pk.collision.RobotCollision = collision_model
    #     self.urdf: yourdfpy.URDF = urdf

    #     # Determine end-effector link
    #     if end_effector_link is None:
    #         # Use last link in kinematic chain
    #         self.ee_link_name = pyroki_robot.links.names[-1]
    #     else:
    #         self.ee_link_name = end_effector_link

    #     self.ee_link_index = pyroki_robot.links.names.index(self.ee_link_name)

    pyroki_robot: pk.Robot
    collision_model: pk.collision.RobotCollision
    urdf: jdc.Static[yourdfpy.URDF]
    end_effector_link: jdc.Static[str]

    @classmethod
    def from_urdf(
        cls, 
        urdf_path: str,
        srdf_path: str,
        mesh_dir: Optional[str] = None,
        end_effector_link: Optional[str] = None,
        default_joint_cfg: Optional[Float[Array, "n"]] = None,
    ) -> "Robot":
        """Load robot from URDF file.

        Args:
            urdf_path: Path to URDF file
            end_effector_link: Name of end-effector link (optional)
            default_joint_cfg: Default joint configuration (optional, defaults to zeros)

        Returns:
            Robot instance
        """
        urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)
        
        # If no default config provided, use zeros as a safe default
        if default_joint_cfg is None:
            # Count actuated joints to determine size
            actuated_joints = [j for j in urdf.joint_map.values() 
                             if j.type in ["revolute", "prismatic", "continuous"]]
            default_joint_cfg = jnp.zeros(len(actuated_joints))
        pyroki_robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_joint_cfg)
        collision_model = pk.collision.RobotCollisionSpherized.from_urdf(urdf, srdf_path=srdf_path)
        if end_effector_link is not None:
            end_effector_link = pyroki_robot.links.names[-1]
        return cls(pyroki_robot, collision_model, urdf, end_effector_link)

    @property
    def num_joints(self) -> int:
        """Number of actuated joints."""
        return self.pyroki_robot.joints.num_actuated_joints

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Names of actuated joints."""
        return self.pyroki_robot.joints.actuated_names
    @property
    def joint_limits_lower(self) -> Float[Array, "n"]:
        """Lower joint limits."""
        return self.pyroki_robot.joints.lower_limits

    @property
    def joint_limits_upper(self) -> Float[Array, "n"]:
        """Upper joint limits."""
        return self.pyroki_robot.joints.upper_limits

    @property
    def velocity_limits(self) -> Float[Array, "n"]:
        """Joint velocity limits."""
        return self.pyroki_robot.joints.velocity_limits

    @property
    def default_config(self) -> Float[Array, "n"]:
        """Default joint configuration (middle of joint limits)."""
        return self.pyroki_robot.joint_var_cls.default_factory()

    def forward_kinematics(
        self, q: Float[Array, "*batch n"]
    ) -> SE3:
        return SE3(self.pyroki_robot.forward_kinematics(q))
    def end_effector_pose(self, q: Float[Array, "*batch n_cfg m_dim"]) -> SE3:
        """Get end-effector pose.

        Args:
            q: Joint configuration(s)

        Returns:
            SE3 pose as (wxyz, xyz) = (w, x, y, z, px, py, pz)
        """
        assert q.shape[-1] == self.num_joints
        poses = self.forward_kinematics(q)
        return SE3(poses.wxyz_xyz[..., -1,:])

    def end_effector_position(self, q: Float[Array, "*batch n"]) -> Float[Array, "*batch 3"]:
        """Get end-effector position.

        Args:
            q: Joint configuration(s)

        Returns:
            Position (x, y, z)
        """
        pose = self.end_effector_pose(q)
        return pose.translation()

    def end_effector_rotation(self, q: Float[Array, "*batch n"]) -> Float[Array, "*batch 4"]:
        """Get end-effector orientation as quaternion.

        Args:
            q: Joint configuration(s)

        Returns:
            Quaternion (w, x, y, z)
        """
        pose = self.end_effector_pose(q)
        return pose.rotation().wxyz

    def compute_self_collision_distance(
        self, q: Float[Array, "*batch n"]
    ) -> Float[Array, "*batch m"]:
        """Compute self-collision distances.

        Args:
            q: Joint configuration(s)

        Returns:
            Array of signed distances (positive = collision free, negative = collision)
        """
        return self.collision_model.compute_self_collision_distance(self.pyroki_robot, q)

    def compute_world_collision_distance(
        self, q: Float[Array, "*batch n"], world_geom: pk.collision.CollGeom
    ) -> Float[Array, "*batch m"]:
        """Compute world collision distances.

        Args:
            q: Joint configuration(s)
            world_geom: World geometry to check against

        Returns:
            Array of signed distances (positive = safe, negative = collision)
        """
        return self.collision_model.compute_world_collision_distance(
            self.pyroki_robot, q, world_geom
        )

    def is_configuration_valid(
        self,
        q: Float[Array, "n"],
        check_limits: bool = True,
        check_self_collision: bool = True,
        collision_margin: float = 0.0,
    ) -> bool:
        """Check if a configuration is valid.

        Args:
            q: Joint configuration
            check_limits: Check joint limits
            check_self_collision: Check self-collisions
            collision_margin: Safety margin for collision checking

        Returns:
            True if configuration is valid
        """
        # Check joint limits
        if check_limits:
            if jnp.any(q < self.joint_limits_lower) or jnp.any(q > self.joint_limits_upper):
                return False

        # Check self-collision
        if check_self_collision:
            distances = self.compute_self_collision_distance(q)
            if jnp.any(distances < collision_margin):
                return False

        return True


    def interpolate_joint_configuration_linear(self, q1: Float[Array, "n"], q2: Float[Array, "n"], n_checks: Int) -> Float[Array, "n"]:
        alphas = jnp.linspace(0, 1, n_checks)
        return q1[None, :] * (1 - alphas[:, None]) + q2[None, :] * alphas[:, None]