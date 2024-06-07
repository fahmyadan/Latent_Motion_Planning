import time
from typing import Callable, List, Optional, Sequence, Tuple, cast

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from .core import Agent, complete_agent_cfg


class Optimizer:
    def __init__(self):
        pass

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution, if necessary.

        Returns:
            (torch.Tensor): the best solution found.
        """
        pass

class MPPIOptimizer(Optimizer):
    """Implements the Model Predictive Path Integral optimization algorithm.

    A derivation of MPPI can be found at https://arxiv.org/abs/2102.09027
    This version is closely related to the original TF implementation used in PDDM with
    some noise sampling modifications and the addition of refinement steps.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        population_size (int): the size of the population.
        gamma (float): reward scaling term.
        sigma (float): noise scaling term used in action sampling.
        beta (float): correlation term between time steps.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        device (torch.device): device where computations will be performed.
    """

    def __init__(
        self,
        num_iterations: int,
        population_size: int,
        gamma: float,
        sigma: float,
        beta: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        device: torch.device,
    ):
        super().__init__()
        self.planning_horizon = len(lower_bound)
        self.population_size = population_size
        self.action_dimension = len(lower_bound[0])
        self.mean = torch.zeros(
            (self.planning_horizon, self.action_dimension),
            device=device,
            dtype=torch.float32,
        )

        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.var = sigma**2 * torch.ones_like(self.lower_bound)
        self.beta = beta
        self.gamma = gamma
        self.refinements = num_iterations
        self.device = device

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of MPPI planner.
        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): Not required
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        for k in range(self.refinements):
            # sample noise and update constrained variances
            noise = torch.empty(
                size=(
                    self.population_size,
                    self.planning_horizon,
                    self.action_dimension,
                ),
                device=self.device,
            )
            noise = mbrl.util.math.truncated_normal_(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.minimum(mv, self.var)
            population = noise.clone() * torch.sqrt(constrained_var)

            # smoothed actions with noise
            population[:, 0, :] = (
                self.beta * (self.mean[0, :] + noise[:, 0, :])
                + (1 - self.beta) * past_action
            )
            for i in range(max(self.planning_horizon - 1, 0)):
                population[:, i + 1, :] = (
                    self.beta * (self.mean[i + 1] + noise[:, i + 1, :])
                    + (1 - self.beta) * population[:, i, :]
                )
            # clipping actions
            # This should still work if the bounds between dimensions are different.
            population = torch.where(
                population > self.upper_bound, self.upper_bound, population
            )
            population = torch.where(
                population < self.lower_bound, self.lower_bound, population
            )
            values = obj_fun(population)
            values[values.isnan()] = -1e-10

            if callback is not None:
                callback(population, values, k)

            # weight actions
            weights = torch.reshape(
                torch.exp(self.gamma * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = population * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

        return self.mean.clone()

class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2)
            .float()
            .to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon

    def optimize(
        self,
        trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (tuple of np.ndarray and float): the best action sequence.
        """
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.previous_solution,
            callback=callback,
        )
        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution.cpu().numpy()

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()


class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True,
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            keep_last_solution=keep_last_solution,
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
            )

        self.optimizer.reset()

    def act(
        self, obs: np.ndarray, optimizer_callback: Optional[Callable] = None, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.
            optimizer_callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (np.ndarray): the action.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences)

            start_time = time.time()
            plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback
            )
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences)

        plan = self.optimizer.optimize(trajectory_eval_fn)
        return plan


def create_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
) -> TrajectoryOptimizerAgent:
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences):

        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent