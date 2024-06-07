#!/bin/bash

eval "$(conda shell.bash hook)"

conda create -n mbrl python=3.10 -y

conda activate mbrl 

pip install poetry

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

poetry install 

cd latent_motion_planning/envs/HighwayEnv || exit

pip install -e . 

