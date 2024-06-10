import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F


import latent_motion_planning.src.utils.math as math

from .model import Ensemble
from .model_utils import EnsembleLinearLayer
from .basic_ensemble import BasicEnsemble
from .model_types import TensorType, TransitionBatch


class PlaNetEnsemble(BasicEnsemble):




    def __init__(self,
        ensemble_size: int,
        device: Union[str, torch.device],
        member_cfg: omegaconf.DictConfig,
        action_size: int, 
        propagation_method: Optional[str] = None,):


        super().__init__(ensemble_size, device, member_cfg, propagation_method)

        self.ensemble_gaussian_params = []
        self.best_model_idx = None 
    
    def update(
        self,
        model_in,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ):
    #TODO: Change update function to update the weights of each member according to their inidividual loss (NLL Loss)
    

        self.train()
        ensemble_loss, ensemble_meta = [], []
        for model in self.members:
            optimizer.zero_grad()
            loss, meta = model.loss(model_in, target)
            ensemble_loss.append(loss.reshape(1,))
            ensemble_meta.append(meta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_norm, norm_type=2)
            if meta is not None:
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                        grad_norm += p.grad.data.norm(2).item() #Use L1 norm instead of l2 to better handle outliers
                    meta["grad_norm"] = grad_norm
            optimizer.step()
        
        self.process_batch_params()
        
        total_loss = torch.concatenate(ensemble_loss).sum()
        # Return the averages amongst the models 
        kin_losses, img_losses, reward_losses, kl_losses = 0.0, 0.0, 0.0, 0.0
        for meta in ensemble_meta:

                kin_losses += meta['kinematic_loss']
                img_losses += meta['img_loss']
                reward_losses += meta['reward_loss']
                kl_losses += meta['kl_loss']
        
        meta.update({"kinematic_loss": kin_losses /self.num_members, 'img_loss': img_losses/self.num_members,\
                     'reward_loss':reward_losses /self.num_members, 'kl_loss': kl_losses /self.num_members})

        return total_loss.item() / self.num_members, meta


    def reset_posterior(self):

        for model in self.members:
            model.reset_posterior()
    
    def update_posterior(self,
        obs: TensorType,
        action: Optional[TensorType] = None,
        rng: Optional[torch.Generator] = None,
   ):

        for model in self.members:
            model.update_posterior(obs, action=action, rng=rng)

    
    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ):

        #TODO: Reset the ensemble properly. Which model state do we use, this needs to be a function of uncertainty of each member in the ensemble. 

        all_states = {}
        for idx, model in enumerate(self.members):

            state_dict = model.reset(obs, rng)
            all_states[idx] = state_dict

        #TODO: Fix return variable. just using a dummy model state for now. We need it to return the latent / belief state of the most certain model. 
                # Use the list of updated gaussian paras (len == num_grad_updates). Use the last update data to determine which model has lowest uncertainty and return that model and state. 
                # return the corresponding model state and model index
        best_idx, _ = self.output_gate()
        self.best_model_idx = best_idx

        return all_states[best_idx]

    def output_gate(self):

        #TODO: Implement a gating mechanism that selects the model with lowest uncertainty for inference 
        latest_params = self.ensemble_gaussian_params[-1]
        latent_size = self.members[0].latent_state_size
        best_var = None 
        best_idx = None 

        for model_idx, params in latest_params.items():
            mean = params[:, :,  : latent_size]
            var = params[:, :, latent_size :]

            total_var = torch.sum(var, dim = 1)
            #Take the average total_time var across the entire batch
            mean_batch_var = torch.mean(total_var, dim = 0) # size = latent_size

            assert torch.any(mean_batch_var >= 0 ), "Cannot have a negative variance"

            if best_var is None or torch.sum(mean_batch_var) < torch.sum(best_var):
                best_var = mean_batch_var 
                best_idx = model_idx  

        assert best_idx is not None, "No best model found. Check the ensemble_gaussian_params."

        return best_idx, best_var  

    def sample(self, act, model_state: Tuple[int, Any], deterministic, rng):

        return self.members[self.best_model_idx].sample(act, model_state, deterministic, rng)
    
    def process_batch_params(self):
        # At the end of each batch, stack the parameters to be (batch, time_horizon, latent_size *2)

        model_params = {}

        for idx, model in enumerate(self.members): 
            with torch.no_grad():
                assert isinstance(model.gaussian_posterior_params, List), "Expected a List of parameters of length = time horizon"
                model_params[idx] = torch.stack(model.gaussian_posterior_params, dim=1)
                model.gaussian_posterior_params = []

        self.ensemble_gaussian_params.append(model_params)

        pass
    

    def count_params(self):

        model_params =[]
        for model in self.members:
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_params.append(pytorch_total_params)
        
        pass








