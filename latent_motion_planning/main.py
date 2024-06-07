

import hydra
import numpy as np
import omegaconf
import torch
import latent_motion_planning.src.algorithm.planet as planet
import mbrl.util.env
import wandb

from latent_motion_planning.src.utils.callbacks import WandbCallback

@hydra.main(config_path="cfgs", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.overrides.get('logging', None) and cfg.overrides.logging.get('wandb', None):
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg)
        wandb_run = wandb.init(project= cfg.overrides.logging.project_name, config=wandb_cfg, sync_tensorboard=True, monitor_gym=True)
        wanb_cbs = [WandbCallback('loss', wandb_run), WandbCallback('reward', wandb_run)]
    else :
        wanb_cbs = None
    if cfg.algorithm.name == "planet":

        if not cfg.overrides.evaluate_trained:
            return planet.train(env, cfg, wandb=wanb_cbs, silent=True)
        else: 
            model_path = cfg.overrides.logging.model_path
            return planet.evaluate_trained_model(model_path, env, cfg)
    else: 
        raise Exception('Unsupported algorithm type')


if __name__ == "__main__":
    run()