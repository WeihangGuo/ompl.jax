from omplx.core.env import JAXEnv, OXMotionValidator, make_state_validity_checker
from omplx.utils.conversion import state_to_jax_array
from jaxtyping import Float, Array
import jax.numpy as jnp

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    import sys
    sys.path.append('./ompl/build/bindings/')
    from ompl import base as ob
    from ompl import geometric as og


class MotionPlanner: 
    def __init__(self, env: JAXEnv, planner:str=['rrtc'], warm_up:bool=True):
        assert planner in ['rrtc']

        self.env = env
        self.dim = env.robot.num_joints

        # Only supports real vector state space for now
        space = ob.RealVectorStateSpace(self.dim)
        bounds = ob.RealVectorBounds(self.dim)
        for i in range(self.dim):
            bounds.setLow(i, float(env.robot.joint_limits_lower[i]))
            bounds.setHigh(i, float(env.robot.joint_limits_upper[i]))
        space.setBounds(bounds)

        self.si = ob.SpaceInformation(space)
        self.si.setStateValidityChecker(make_state_validity_checker(env))
        self.si.setMotionValidator(OXMotionValidator(self.si, env))
        self.si.setup()
        self.ss = og.SimpleSetup(self.si)
        if planner == 'rrtc':
            self.planner = og.RRTConnect(self.si)
        self.ss.setPlanner(self.planner)
        self.planner_cache_empty = True

        if warm_up:
            print(f"Dry running planner to warm up...")
            self.plan(start=env.robot.pyroki_robot.joint_var_cls.default_factory(), 
                goal=env.robot.pyroki_robot.joint_var_cls.default_factory())
            self.planner.clearQuery()
            print(f"Done! Warm up complete.")
    def _clean_planner_cache(self):
        if not self.planner_cache_empty:
            self.planner.clearQuery()
            self.planner_cache_empty = True

    def plan(self, start: Float[Array, "n"], goal: Float[Array, "n"]) -> Float[Array, "n"]:
        assert start.shape == goal.shape == (self.dim,)
        self._clean_planner_cache()
        start_state = self.si.allocState()
        goal_state = self.si.allocState()
        for i in range(self.dim):
            start_state[i] = float(start[i])
            goal_state[i] = float(goal[i])
        self.ss.setStartAndGoalStates(start_state, goal_state)
        solved = self.ss.solve(10.0)
        self.planner_cache_empty = False
        if solved:
            path = self.ss.getSolutionPath()
            path.interpolate()
            return path
        else:
            return None
    def path_to_jax_array(self, path: ob.Path) -> Float[Array, "n_states n_joints"]:
        return jnp.array([state_to_jax_array(state, self.dim) for state in path.getStates()])