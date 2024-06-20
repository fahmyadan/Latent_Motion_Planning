# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import pathlib
from typing import List, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from latent_motion_planning.src.utils.env_fns import no_termination
from latent_motion_planning.src.models import ModelEnv, ModelTrainer, Model, PlaNetModel, PlaNetEnsemble
from latent_motion_planning.src.planners import RandomAgent, create_trajectory_optim_agent_for_model, MPPIAgent
# from mbrl.util import Logger
from latent_motion_planning.src.utils.common import (
    create_replay_buffer,
    get_sequence_buffer_iterator,
    rollout_agent_trajectories,
    MonitorKPIs
)

METRICS_LOG_FORMAT = [
    ("observations_loss", "OL", "float"),
    ("reward_loss", "RL", "float"),
    ("gradient_norm", "GN", "float"),
    ("kl_loss", "KL", "float"),
]
import matplotlib.pyplot as plt
import wandb as wb
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np 
episode_monitor = 50 

def train(
    env: gym.Env,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Union[Optional[str], pathlib.Path] = None,
    wandb = None 
) -> np.float32:
    # Experiment initialization
    debug_mode = cfg.get("debug_mode", False)

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = pathlib.Path(work_dir)
    print(f"Results will be saved at {work_dir}.")

    # if silent:
    #     logger = None
    # else:
    #     logger = Logger(work_dir)
    #     logger.register_group("metrics", METRICS_LOG_FORMAT, color="yellow")
    #     logger.register_group(
    #         mbrl.constants.RESULTS_LOG_NAME,
    #         [
    #             ("env_step", "S", "int"),
    #             ("train_episode_reward", "RT", "float"),
    #             ("episode_reward", "ET", "float"),
    #         ],
    #         color="green",
    #     )

    rng = torch.Generator(device=cfg.device)
    rng.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(seed=cfg.seed)

    # Create replay buffer and collect initial data
    if not isinstance(env.observation_space, gym.spaces.tuple.Tuple):
        replay_buffer = create_replay_buffer(
            cfg,
            env.observation_space.shape,
            env.action_space.shape,
            collect_trajectories=True,
            rng=np_rng,
        )
    else: 
        tuple_space = [obs_space.shape for obs_space in env.observation_space.spaces]

        replay_buffer = create_replay_buffer(
            cfg,
            tuple_space,
            env.action_space.shape,
            collect_trajectories=True,
            rng=np_rng,
            tuple_obs= True
        )

    
    total_demo_rewards , metrics = rollout_agent_trajectories(
        env,
        cfg.algorithm.num_initial_trajectories,
        # MPPIAgent(env, **omegaconf.OmegaConf.to_container(cfg.overrides.mppi)),
        RandomAgent(env),
        agent_kwargs={},
        replay_buffer=replay_buffer,
        collect_full_trajectories=True,
        trial_length=cfg.overrides.trial_length,
        agent_uses_low_dim_obs=False,
        kwargs=cfg.overrides.kpis
    )

    # Create PlaNet model
    cfg.dynamics_model.action_size = env.action_space.shape[0]
    planet = hydra.utils.instantiate(cfg.dynamics_model)
    assert isinstance(planet, PlaNetModel) or isinstance(planet, PlaNetEnsemble), f"expected dynamics model of type {PlaNetModel} or {PlaNetEnsemble}"
    model_env = ModelEnv(env, planet, no_termination, generator=rng)
    trainer = ModelTrainer(planet, cfg.overrides, logger=None, optim_lr=1e-3, optim_eps=1e-4)

    # Create MPPI agent
    # This agent rolls outs trajectories using ModelEnv, which uses planet.sample()
    # to simulate the trajectories from the prior transition model
    # The starting point for trajectories is conditioned on the latest observation,
    # for which we use planet.update_posterior() after each environment step
    agent = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)

    # Callback and containers to accumulate training statistics and average over batch
    rec_losses: List[float] = []
    kin_losses = []
    reward_losses: List[float] = []
    kl_losses: List[float] = []
    grad_norms: List[float] = []

    def get_metrics_and_clear_metric_containers():
        metrics_ = {
            "observations_loss": np.mean(rec_losses).item(),
            "reward_loss": np.mean(reward_losses).item(),
            "gradient_norm": np.mean(grad_norms).item(),
            "kl_loss": np.mean(kl_losses).item(),
        }

        for c in [rec_losses, reward_losses, kl_losses, grad_norms]:
            c.clear()

        return metrics_

    def batch_callback(train_episode, total_grad_updates, grad_updates, _loss, meta, _mode):
        if meta:
            rec_losses.append(meta["img_loss"])
            kin_losses.append(meta["kinematic_loss"])
            reward_losses.append(meta["reward_loss"])
            kl_losses.append(meta["kl_loss"])
            if train_episode % episode_monitor == 0 and grad_updates % int(total_grad_updates -1) == 0:
                log_meta(meta, train_episode, grad_updates)
                
            if "grad_norm" in meta:
                grad_norms.append(meta["grad_norm"])

    def _denormalise_pixel_obs(obs: torch.Tensor) -> np.ndarray:

         # Reverse the normalization
        obs = (obs + 0.5) * 256.0
        return obs.astype(np.uint8)


    
    def log_meta(meta, _epoch, _update):
        from PIL import Image
        import random
        import os
        from pathlib import Path 
        global recons_dir
        recons_dir= str(Path(__file__).parents[3]) + '/recons' #/ 256.0 - 0.5

        if not os.path.exists(recons_dir):
                os.makedirs(recons_dir)
        
        reconstruction = meta['img_reconstruction'].cpu().numpy()
        target_obs = meta['target_obs_img'].cpu().numpy()
        target_obs = _denormalise_pixel_obs(target_obs)
        reconstruction = _denormalise_pixel_obs(reconstruction)
        rand_batch = random.randint(0,reconstruction.shape[0] - 1)
        horizon = reconstruction.shape[1]
        # Pick a random batch from the training epoch reconstructions and targets

        for t in range(horizon):

            seq_t = reconstruction[rand_batch, t]
            target_t = target_obs[rand_batch,t]
            seq_rgb = np.clip(seq_t, 0, 255).astype(np.uint8)

            if seq_t.shape[0] == 1: 
                seq_t = np.squeeze(seq_t, axis=0)
                seq_t = (seq_t * 255).astype(np.uint8)
            else:
                seq_t = np.transpose(seq_t, (1, 2, 0)).astype(np.uint8)
                seq_t = np.clip(seq_t, 0, 255)

                #Denormalise the taregt image (see def _process_pixel_observations)
                # denorm_target = (target_t +0.5) * 256.0
                # denorm_target = denorm_target.astype(np.uint8)
                target_t = np.transpose(target_t, (1, 2, 0)).astype(np.uint8)
                # target_t = np.clip(target_t * 255, 0, 255).astype(np.uint8)


            # im = Image.fromarray(seq_t[:,:, -1])
            # im_target = Image.fromarray(target_t)
            im = Image.fromarray(seq_t)
            im_target = Image.fromarray(target_t)
            
            im.save(os.path.join(recons_dir, f'reconstruction_epoch_{_epoch}_grad_{_update}_t_{t}_.png'))
            im_target.save(os.path.join(recons_dir, f'target_epoch_{_epoch}_grad_{_update}_t_{t}_.png'))


    def is_test_episode(episode_):
        return episode_ % cfg.algorithm.test_frequency == 0
    
    def episode_trigger(episode_id):
        return episode_id % episode_monitor == 0
    if env.render_mode =='rgb_array':
        if wandb:
            vid_dir = wandb[0].dir + '/videos'
            env = RecordVideo(env, video_folder=vid_dir, episode_trigger=episode_trigger)

    # PlaNet loop
    step = replay_buffer.num_stored
    total_rewards = 0.0
    n_epochs = cfg.overrides.n_epochs
    for episode in tqdm(range(cfg.algorithm.num_episodes)):
        # Train the model for one epoch of `num_grad_updates`
        dataset, _ = get_sequence_buffer_iterator(
            replay_buffer,
            cfg.overrides.batch_size,
            0,  # no validation data
            cfg.overrides.sequence_length,
            max_batches_per_loop_train=cfg.overrides.num_grad_updates,
            use_simple_sampler=True,
        )
        if wandb:
            trainer.train(
                dataset, num_epochs=n_epochs, batch_callback=batch_callback, evaluate=False, callback=wandb[0], episode=episode
            )
        else:
            trainer.train(
                dataset, num_epochs=n_epochs, batch_callback=batch_callback, evaluate=False, episode=episode
            )
        if cfg.overrides.logging.wandb:
            work_dir = pathlib.Path(wandb[0].dir)
            planet.save(work_dir)
        if cfg.overrides.get("save_replay_buffer", False):
            replay_buffer.save(work_dir)
        metrics = get_metrics_and_clear_metric_containers()
        # if logger: 
        #     logger.log_data("metrics", metrics)

        # Collect one episode of data
        episode_reward = 0.0
        obs, _ = env.reset()
        agent.reset()
        planet.reset_posterior()
        action = None
        terminated = False
        truncated = False
        pbar = tqdm(total=1000)
        action_repeat = 1
        while not terminated and not truncated:
            planet.update_posterior(obs, action=action, rng=rng)
            action_noise = (
                0
                if is_test_episode(episode)
                else cfg.overrides.action_noise_std
                * np_rng.standard_normal(env.action_space.shape[0])
            )
            action = agent.act(obs) + action_noise
            action = np.clip(
                action, -1.0, 1.0, dtype=env.action_space.dtype
            )  # to account for the noise and fix dtype
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            # replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
            # episode_reward += reward
            # obs = next_obs
            for _ in range(action_repeat):
                if terminated or truncated:
                    break
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                obs = next_obs
                step += 1

            if debug_mode:
                print(f"step: {step}, reward: {reward}.")
            step += 1
            pbar.update(1)
        pbar.close()
        total_rewards += episode_reward
        # logger.log_data(
        #     mbrl.constants.RESULTS_LOG_NAME,
        #     {
        #         "episode_reward": episode_reward * is_test_episode(episode),
        #         "train_episode_reward": episode_reward * (1 - is_test_episode(episode)),
        #         "env_step": step,
        #     },
        # )
        if wandb: 
            # reward_cb = wandb[1]
            # ep_reward = episode_reward * is_test_episode(episode)
            # train_ep_reward = episode_reward * (1 - is_test_episode(episode))
            # reward_cb(None, None, None, ep_reward, train_ep_reward, episode)
            if is_test_episode(episode):
                reward_cb = wandb[2]
                reward_cb(None, None, None, episode_reward, None, episode)
            else:
                reward_cb = wandb[1]
                reward_cb(None, None, None, None, episode_reward, episode)
            if env.render_mode == 'rgb_array' and episode_trigger(episode):
                all_imgs = os.listdir(recons_dir)
                video_path = f'{vid_dir}/rl-video-episode-{episode}.mp4'
                # wb.log({"Intersection": wb.Video(video_path, fps=4, format="mp4")})

                log_recon_imgs(all_imgs, episode)


    # returns average episode reward (e.g., to use for tuning learning curves)
    return total_rewards / cfg.algorithm.num_episodes

def log_recon_imgs(imgs : List, train_episode: int):
    
    logged_images = []

    for idx, img in enumerate(imgs):
        splits = img.split('_')

        if int(splits[2]) == train_episode:
            log_file_path = recons_dir  +'/' + img
            logged_images.append(wb.Image(log_file_path, caption=img))

    if logged_images:
        wb.log({"reconstructed_images": logged_images})
   
    pass




def evaluate_trained_model(model_path, env, cfg):

    rng = torch.Generator(device=cfg.device)
    rng.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(seed=cfg.seed)

    cfg.dynamics_model.action_size = env.action_space.shape[0]
    planet = hydra.utils.instantiate(cfg.dynamics_model)
    assert isinstance(planet, mbrl.models.PlaNetModel)

    planet.load_state_dict(torch.load(model_path))
    planet.eval()
    model_env = ModelEnv(env, planet, no_termination, generator=rng)

    n_episodes = cfg.overrides.logging.eval_episodes
    agent = create_trajectory_optim_agent_for_model(model_env, cfg.algorithm.agent)
    total_rewards = []
    pbar = tqdm(total=n_episodes)
    action_repeat = 1
    # Collect one episode of data
    for eval_episode in range(n_episodes):
        episode_reward = 0.0
        obs, _ = env.reset()
        agent.reset()
        planet.reset_posterior()
        action = None
        terminated = False
        truncated = False
        step = 0
        while not terminated and not truncated:
            planet.update_posterior(obs, action=action, rng=rng)
            action_noise = 0
            action = agent.act(obs) + action_noise
            action = np.clip(
                action, -1.0, 1.0, dtype=env.action_space.dtype
            )  # to account for the noise and fix dtype
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            # episode_reward += reward
            # obs = next_obs

            for _ in range(action_repeat):
                if terminated or truncated:
                    break
                next_obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                obs = next_obs
                step += 1
            #TODO: Add some KPI logging, travel time collision etc. 
            # if debug_mode:
            #     print(f"step: {step}, reward: {reward}.")
            step += 1
        pbar.update(1)
        
        total_rewards.append(episode_reward)
        # obs, _ = env.reset()
        # terminated = False
        # truncated = False
    pbar.close()

    avg_reward = np.mean(total_rewards)
    fig, ax = plt.subplots()

    # Generate a boxplot
    ax.boxplot(total_rewards, showmeans= True)
    # ax.axhline(y=avg_reward, color='r', linestyle='-', label='Average Reward')

    # Set the title and labels as needed
    ax.set_title('Boxplot of Rewards')
    ax.set_ylabel('Values')
    ax.legend()
    # Show the plot
    plt.show()

    pass
