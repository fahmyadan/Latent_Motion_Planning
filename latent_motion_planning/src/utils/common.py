import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import hydra
import numpy as np
import omegaconf

import latent_motion_planning.src.models
import latent_motion_planning.src.planners


from .replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    SequenceTransitionIterator,
    SequenceTransitionSampler,
    TransitionIterator,
    TupleReplay
)
import highway_env



def create_replay_buffer(
    cfg: omegaconf.DictConfig,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    obs_type: Type = np.float32,
    action_type: Type = np.float32,
    reward_type: Type = np.float32,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
    tuple_obs: bool = False
) -> ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -num_steps (int, optional): how many steps to take in the environment
            -trial_length (int, optional): the maximum length for trials. Only needed if
                ``collect_trajectories == True``.

    The size of the replay buffer can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing ``cfg.overrides.num_steps``.
    Specifying dataset set size directly takes precedence over number of steps.

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (Sequence of ints): the shape of observation arrays.
        act_shape (Sequence of ints): the shape of action arrays.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (:class:`mbrl.replay_buffer.ReplayBuffer`): the replay buffer.
    """
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
    if not dataset_size:
        dataset_size = cfg.overrides.num_steps
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.overrides.trial_length is None:
            raise ValueError(
                "cfg.overrides.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len = cfg.overrides.trial_length

    if tuple_obs:
        replay_buffer = TupleReplay(
            dataset_size,
            obs_shape,
            act_shape,
            obs_type=obs_type,
            action_type=action_type,
            reward_type=reward_type,
            rng=rng,
            max_trajectory_length=maybe_max_trajectory_len,
        )
    else:
        replay_buffer = ReplayBuffer(
            dataset_size,
            obs_shape,
            act_shape,
            obs_type=obs_type,
            action_type=action_type,
            reward_type=reward_type,
            rng=rng,
            max_trajectory_length=maybe_max_trajectory_len,
        )
    

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer

def get_basic_buffer_iterators(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    bootstrap_permutes: bool = False,
) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        ensemble_size (int): the size of the ensemble being trained.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
            the bootstrap data using permutations of the original data. Otherwise
            it will use sampling with replacement. Defaults to ``False``.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """
    data = replay_buffer.get_all(shuffle=True)
    val_size = int(replay_buffer.num_stored * val_ratio)
    train_size = replay_buffer.num_stored - val_size
    train_data = data[:train_size]
    train_iter = BootstrapIterator(
        train_data,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=replay_buffer.rng,
    )

    val_iter = None
    if val_size > 0:
        val_data = data[train_size:]
        val_iter = TransitionIterator(
            val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
        )

    return train_iter, val_iter


_SequenceIterType = Union[SequenceTransitionIterator, SequenceTransitionSampler]

def get_sequence_buffer_iterator(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    sequence_length: int,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    max_batches_per_loop_train: Optional[int] = None,
    max_batches_per_loop_val: Optional[int] = None,
    use_simple_sampler: bool = False,
) -> Tuple[_SequenceIterType, Optional[_SequenceIterType]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        sequence_length (int): the length of the sequences returned by the iterators.
        ensemble_size (int): the number of models in the ensemble.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        max_batches_per_loop_train (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the training iterator.
        max_batches_per_loop_val (int, optional): if given, specifies how many batches
            to return (at most) over a full loop of the validation iterator.
        use_simple_sampler (int): if ``True``, returns an iterator of type
            :class:`mbrl.replay_buffer.SequenceTransitionSampler` instead of
            :class:`mbrl.replay_buffer.SequenceTransitionIterator`.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """

    assert replay_buffer.stores_trajectories, (
        "The passed replay buffer does not store trajectory information. "
        "Make sure that the replay buffer is created with the max_trajectory_length "
        "parameter set."
    )

    transitions = replay_buffer.get_all()
    num_trajectories = len(replay_buffer.trajectory_indices)
    val_size = int(num_trajectories * val_ratio)
    train_size = num_trajectories - val_size
    all_trajectories = replay_buffer.rng.permutation(replay_buffer.trajectory_indices)
    train_trajectories = all_trajectories[:train_size]

    if use_simple_sampler:
        train_iterator: _SequenceIterType = SequenceTransitionSampler(
            transitions,
            train_trajectories,  # type:ignore
            batch_size,
            sequence_length,
            max_batches_per_loop_train,
            rng=replay_buffer.rng,
        )
    else:
        train_iterator = SequenceTransitionIterator(
            transitions,
            train_trajectories,  # type: ignore
            batch_size,
            sequence_length,
            ensemble_size,
            shuffle_each_epoch=shuffle_each_epoch,
            rng=replay_buffer.rng,
            max_batches_per_loop=max_batches_per_loop_train,
        )

    val_iterator: Optional[_SequenceIterType] = None
    if val_size > 0:
        val_trajectories = all_trajectories[train_size:]
        if use_simple_sampler:
            val_iterator = SequenceTransitionSampler(
                transitions,
                val_trajectories,  # type: ignore
                batch_size,
                sequence_length,
                max_batches_per_loop_val,
                rng=replay_buffer.rng,
            )
        else:
            val_iterator = SequenceTransitionIterator(
                transitions,
                val_trajectories,  # type: ignore
                batch_size,
                sequence_length,
                1,
                shuffle_each_epoch=shuffle_each_epoch,
                rng=replay_buffer.rng,
                max_batches_per_loop=max_batches_per_loop_val,
            )
            val_iterator.toggle_bootstrap()

    return train_iterator, val_iterator


def rollout_agent_trajectories(
    env: gym.Env,
    steps_or_trials_to_collect: int,
    agent: latent_motion_planning.src.planners.Agent,
    agent_kwargs: Dict,
    trial_length: Optional[int] = None,
    callback: Optional[Callable] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    collect_full_trajectories: bool = False,
    agent_uses_low_dim_obs: bool = False,
    seed: Optional[int] = None,
    **kwargs
) -> List[float]:
    """Rollout agent trajectories in the given environment.

    Rollouts trajectories in the environment using actions produced by the given agent.
    Optionally, it stores the saved data into a replay buffer.

    Args:
        env (gym.Env): the environment to step.
        steps_or_trials_to_collect (int): how many steps of the environment to collect. If
            ``collect_trajectories=True``, it indicates the number of trials instead.
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        trial_length (int, optional): a maximum length for trials (env will be reset regularly
            after this many number of steps). Defaults to ``None``, in which case trials
            will end when the environment returns ``terminated=True`` or ``truncated=True``.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, terminated, truncated)`.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`, optional):
            a replay buffer to store data to use for training.
        collect_full_trajectories (bool): if ``True``, indicates that replay buffers should
            collect full trajectories. This only affects the split between training and
            validation buffers. If ``collect_trajectories=True``, the split is done over
            trials (full trials in each dataset); otherwise, it's done across steps.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper` and replay_buffer is not ``None``.
            If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (list(float)): Total rewards obtained at each complete trial.
    """
    if (
        replay_buffer is not None
        and replay_buffer.stores_trajectories
        and not collect_full_trajectories
    ):
        # Might be better as a warning but it's possible that users will miss it.
        raise RuntimeError(
            "Replay buffer is tracking trajectory information but "
            "collect_trajectories is set to False, which will result in "
            "corrupted trajectory data."
        )

    
    trial = 0
    kpis = [vals for vals in kwargs.values()]
    monitor = MonitorKPIs(kpis[0], env)
    total_rewards: List[float] = []
    while True:
        step = 0
        obs, info = env.reset(seed=seed)
        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            if replay_buffer is not None:
                next_obs, reward, terminated, truncated, _ = step_env_and_add_to_buffer(
                    env,
                    obs,
                    agent,
                    agent_kwargs,
                    replay_buffer,
                    callback=callback,
                    agent_uses_low_dim_obs=agent_uses_low_dim_obs,
                )
            else:
                if agent_uses_low_dim_obs:
                    raise RuntimeError(
                        "Option agent_uses_low_dim_obs is only valid if a "
                        "replay buffer is given."
                    )
                action = agent.act(obs, **agent_kwargs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                if callback:
                    callback((obs, action, next_obs, reward, terminated, truncated))
            obs = next_obs
            total_reward += reward
            step += 1
            if not collect_full_trajectories and step == steps_or_trials_to_collect:
                total_rewards.append(total_reward)
                return total_rewards
            if trial_length and step % trial_length == 0:
                if (
                    collect_full_trajectories
                    and not terminated
                    and replay_buffer is not None
                ):
                    replay_buffer.close_trajectory()
                break
        trial += 1
        total_rewards.append(total_reward)
        monitor.all_rewards.append(total_reward)  
        monitor.monitor()
        if collect_full_trajectories and trial == steps_or_trials_to_collect:
            break
        
    return total_rewards, monitor

def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    agent: latent_motion_planning.src.planners.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
    callback: Optional[Callable] = None,
    agent_uses_low_dim_obs: bool = False,
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Steps the environment with an agent's action and populates the replay buffer.

    Args:
        env (gym.Env): the environment to step.
        obs (np.ndarray): the latest observation returned by the environment (used to obtain
            an action from the agent).
        agent (:class:`mbrl.planning.Agent`): the agent used to generate an action.
        agent_kwargs (dict): any keyword arguments to pass to `agent.act()` method.
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer
            containing stored data.
        callback (callable, optional): a function that will be called using the generated
            transition data `(obs, action. next_obs, reward, terminated, truncated)`.
        agent_uses_low_dim_obs (bool): only valid if env is of type
            :class:`mbrl.env.MujocoGymPixelWrapper`. If ``True``, instead of passing the obs
            produced by env.reset/step to the agent, it will pass
            obs = env.get_last_low_dim_obs(). This is useful for rolling out an agent
            trained with low dimensional obs, but collect pixel obs in the replay buffer.

    Returns:
        (tuple): next observation, reward, terminated, truncated and meta-info, respectively,
        as generated by `env.step(agent.act(obs))`.
    """

    if agent_uses_low_dim_obs and not hasattr(env, "get_last_low_dim_obs"):
        raise RuntimeError(
            "Option agent_uses_low_dim_obs is only compatible with "
            "env of type mbrl.env.MujocoGymPixelWrapper."
        )
    if agent_uses_low_dim_obs:
        agent_obs = getattr(env, "get_last_low_dim_obs")()
    else:
        agent_obs = obs
    action = agent.act(agent_obs, **agent_kwargs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
    if callback:
        callback(*(obs, action, next_obs, reward, terminated, truncated))
    return next_obs, reward, terminated, truncated, info


class MonitorKPIs():

    def __init__(self, kpis, env):

        self.travel_time, self.all_tt = kpis['travel_time'], []
        self.delay, self.all_delay = kpis['delay'], []
        self.collisions, self.all_collisions = kpis['collisions'], []
        self.reward, self.all_rewards = kpis['reward'], []
        self.arrived = []
        self.off_road = []
        self.truncated_time = []

        self.env = env.unwrapped
        max_speed = self.env.controlled_vehicles[0].MAX_SPEED
        total_dist = 38.77302896 + 20.420352248334 + 25
        self.unimpeded_time = total_dist / max_speed * self.env.config['simulation_frequency']
       

    def monitor(self):

        collisions = self.check_collision()
        arrived = self.check_arrived()
        off_road = self.check_off_road()

        self.arrived.append(arrived)
        self.all_collisions.append(collisions)
        self.off_road.append(off_road)

        self.calculate_tt()
        self.calculate_delay()

        if self.env._is_truncated():
            self.truncated_time.append(1)
        else:
            self.truncated_time.append(0)
        
        
 
    def calculate_delay(self):

        """
        The delay is the difference between actual travel time and unimpeded travel time 

        Assume v_constant = 40m/s 
                total travel distance = 38.773 (straight lane 1 - start pos) + 20.420 (circular lane length) + 25m (exit length) 
        """

        veh_delay = []

        for veh in self.env.controlled_vehicles:
            if self.env.has_arrived(veh):
                tt = self.env.time * self.env.config['simulation_frequency'] 
                delay = tt - self.unimpeded_time
            else: 
                delay = None 
            veh_delay.append(delay)
        self.all_delay.append(veh_delay)

        pass
    def calculate_tt(self):

        veh_tt = []

        for veh in self.env.controlled_vehicles:
            if self.env.has_arrived(veh):
                tt = self.env.time * self.env.config['simulation_frequency'] #Confirm if this is correct 
            else:
                tt = None
            veh_tt.append(tt)
        
        self.all_tt.append(veh_tt)
            


    def check_collision(self):

        collisions = []
        for veh in self.env.controlled_vehicles:

            if veh.crashed:
                collisions.append(1)
            else:
                collisions.append(0)

        return collisions
    
    def check_arrived(self):

        has_arrived = self.env.has_arrived
        arrived = []
        for veh in self.env.controlled_vehicles:

            if has_arrived(veh):

                arrived.append(1)
            else:
                arrived.append(0)
        return arrived
    
    def check_off_road(self):
        off_road = []
        for veh in self.env.controlled_vehicles:

            if not veh.on_road:
                off_road.append(1)
            else:
                off_road.append(0)

        return off_road