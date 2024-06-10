# Background

This repository is a library for Model Based Reinoforcement Learning (MBRL) building on [this paper](https://arxiv.org/abs/2104.10159). 

It is concerned with Autonomous Driving environments, where the goal is to learn the transition dynamics of the environment in closed loop: 

$$
P_{\theta}(x_{t+1}| x_t, u_t) \sim N(\mu, \sigma^2)
$$

In model based planning cases where we cannot access the ground truth dynamics, this approximation can be used for multi-step planning. Examples include RL (Policy Optimization, Q-Learning etc.) or sampling based planning (e.g. MPPI, CEM etc.).

This is an implementation of the paper [Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).

# Getting Started 
## Requirements
 - Python 3.10+ 
 - Conda & Pip 
 - (Optional) CUDA > 12.0 

It will still run without CUDA but a large ensemble size will have larger memory requirements. 

We have included a simple installation script:

```bash
git clone --recurse-submodules https://github.com/fahmyadan/Latent_Motion_Planning.git 
sudo chmod +x setup.bash 
./setup.bash
conda activate z_plan
python3 latent_motion_planning/main.py algorithm=planet overrides=planet_highway_env dynamics_model=planet_hw action_optimizer=mppi
```

To run the ensemble networks: 

```bash
conda activate z_plan
python3 latent_motion_planning/main.py algorithm=planet overrides=planet_highway_env dynamics_model=planet_ensemble action_optimizer=mppi

```

