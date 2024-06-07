# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import pathlib
from typing import Any, Union

import gymnasium as gym
import hydra
import numpy as np
import omegaconf

import latent_motion_planning.src.models
from pytorch_mppi import MPPI
import torch 
from highway_env.road.lane import StraightLane, CircularLane



class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass


class RandomAgent(Agent):
    """An agent that samples action from the environments action space.

    Args:
        env (gym.Env): the environment on which the agent will act.
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, *_args, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Returns:
            (np.ndarray): an action sampled from the environment's action space.
        """
        return self.env.action_space.sample()

class MPPIAgent(Agent):
    def __init__(self, env: gym.Env, **kwargs):
        self.env = env.unwrapped
        self.LENGTH = self.env.controlled_vehicles[0].LENGTH
        self.dt = 1 / self.env.config['simulation_frequency']
        self.mppi_args = kwargs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mppi_args['noise_sigma'] = torch.tensor([self.mppi_args['noise_sigma'][0], np.radians(self.mppi_args['noise_sigma'][1])], 
                                                    device = self.device, dtype= torch.double)

        self.mppi_args['noise_mu'] =  torch.tensor(self.mppi_args['noise_mu'], device = self.device, dtype= torch.double)
        self.mppi_args['u_min'] = torch.tensor(self.mppi_args['u_min'])
        self.mppi_args['u_max'] = torch.tensor(self.mppi_args['u_max'])
        self.mppi_args['device'] = self.device
        self.mppi = MPPI(self.dynamics_f, self.running_cost, **self.mppi_args)
        self.veh_obj = self.env.controlled_vehicles[0]

        self.t = 0 

        

    def dynamics_f(self, state, action):
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        yaw = state[:, 2].view(-1, 1)
        v = state[:, 3].view(-1, 1)


        acc = action[:, 0].view(-1, 1)
        steer = action[:, 1].view(-1, 1)

        wheel_base = self.LENGTH /2 
        dt = self.dt 

        new_speed = v  + acc * dt

        new_x = x + (v * torch.cos(yaw) * dt)
        new_y = y + (v * torch.sin(yaw) * dt)
        new_yaw = yaw + v / wheel_base * torch.tan(steer) * dt

        next_state = torch.cat((new_x, new_y, new_yaw, new_speed), dim= 1)

        return next_state 

    def running_cost(self, state, action):

        lanes_on_route = [self.veh_obj.road.network.get_lane(r) for r in self.veh_obj.route]
        trajectory = torch.tensor(self.global_path(lanes_on_route)).to(self.device)

        dev_cost = self.deviation(state[:,:2], trajectory)

        return 1000 * dev_cost **2

    def deviation(self, states, trajectory):
        #for each k, get closest_point in trajectory. closest (k, 1) where column is distance 
        samples, n = states.shape
        states = states.reshape(samples, 1, n)

        distance = torch.norm(states - trajectory, dim= 2)

        min_distance = torch.min(distance, dim = 1)

        return min_distance.values
    def global_path(self, lanes_to_go):

        trajectory = []

        for lane in lanes_to_go: 
            if isinstance(lane, CircularLane):

                arc_points = lane.get_arc_points()
                nodeless_arc_points = arc_points[1:-1]
                trajectory.append(nodeless_arc_points)
            elif isinstance(lane, StraightLane): 
                startx, starty = lane.start[0], lane.start[1]
                endx, endy = lane.end[0], lane.end[1]
                x_n = np.linspace(startx, endx, 40)
                y_n = np.linspace(starty, endy, 40)
                new = np.column_stack((x_n, y_n))
                nodeless_new = new[1:-1]
                trajectory.append(nodeless_new)

        trajectory = np.concatenate(trajectory)
        closest_pos = trajectory[0]
        for idx, pos in enumerate(trajectory):

            if np.linalg.norm(self.veh_obj.position - pos) <= np.linalg.norm(self.veh_obj.position - closest_pos):
                closest_pos = pos
                closest_idx = idx

        return trajectory[closest_idx:]


    def act(self, obs, **_kwargs) -> np.ndarray:
        self.t +=1 
    
        position = self.env.unwrapped.controlled_vehicles[0].position
        heading = np.array(self.env.unwrapped.controlled_vehicles[0].heading).reshape(1,)
        speed = np.array(self.env.unwrapped.controlled_vehicles[0].speed).reshape(1,)

        state = np.concatenate([position, heading, speed])

        # self.veh_obj.follow_road() 

        if self.t % 25 == 0 : 
            return  self.env.action_space.sample()
     
        action = self.mppi.command(state)
        
        return action.cpu().numpy()

    




def complete_agent_cfg(
    env: Union[gym.Env], agent_cfg: omegaconf.DictConfig
):
    """Completes an agent's configuration given information from the environment.

    The goal of this function is to completed information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "action_dim": set to env.action_space.shape
        - "action_range": set to max(env.action_space.high) - min(env.action_space.low)
        - "action_lb": set to env.action_space.low
        - "action_ub": set to env.action_space.high

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.

    """
    if not isinstance(env.observation_space, gym.spaces.tuple.Tuple):
        obs_shape = env.observation_space.shape
    else:
        obs_shape = env.observation_space.spaces[1].shape
    act_shape = env.action_space.shape

    def _check_and_replace(key: str, value: Any, cfg: omegaconf.DictConfig):
        if key in cfg.keys() and key not in cfg:
            setattr(cfg, key, value)

    _check_and_replace("num_inputs", obs_shape[0], agent_cfg)
    if "action_space" in agent_cfg.keys() and isinstance(
        agent_cfg.action_space, omegaconf.DictConfig
    ):
        _check_and_replace("low", env.action_space.low.tolist(), agent_cfg.action_space)
        _check_and_replace(
            "high", env.action_space.high.tolist(), agent_cfg.action_space
        )
        _check_and_replace("shape", env.action_space.shape, agent_cfg.action_space)

    if "obs_dim" in agent_cfg.keys() and "obs_dim" not in agent_cfg:
        agent_cfg.obs_dim = obs_shape[0]
    if "action_dim" in agent_cfg.keys() and "action_dim" not in agent_cfg:
        agent_cfg.action_dim = act_shape[0]
    if "action_range" in agent_cfg.keys() and "action_range" not in agent_cfg:
        agent_cfg.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
    if "action_lb" in agent_cfg.keys() and "action_lb" not in agent_cfg:
        agent_cfg.action_lb = env.action_space.low.tolist()
    if "action_ub" in agent_cfg.keys() and "action_ub" not in agent_cfg:
        agent_cfg.action_ub = env.action_space.high.tolist()

    return agent_cfg


def load_agent(agent_path: Union[str, pathlib.Path], env: gym.Env) -> Agent:
    """Loads an agent from a Hydra config file at the given path.

    For agent of type "pytorch_sac.agent.sac.SACAgent", the directory
    must contain the following files:

        - ".hydra/config.yaml": the Hydra configuration for the agent.
        - "critic.pth": the saved checkpoint for the critic.
        - "actor.pth": the saved checkpoint for the actor.

    Args:
        agent_path (str or pathlib.Path): a path to the directory where the agent is saved.
        env (gym.Env): the environment on which the agent will operate (only used to complete
            the agent's configuration).

    Returns:
        (Agent): the new agent.
    """
    agent_path = pathlib.Path(agent_path)
    cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")

    if cfg.algorithm.agent._target_ == "mbrl.third_party.pytorch_sac_pranz24.sac.SAC":
        import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac

        from .sac_wrapper import SACAgent

        complete_agent_cfg(env, cfg.algorithm.agent)
        agent: pytorch_sac.SAC = hydra.utils.instantiate(cfg.algorithm.agent)
        agent.load_checkpoint(ckpt_path=agent_path / "sac.pth")
        return SACAgent(agent)
    else:
        raise ValueError("Invalid agent configuration.")
