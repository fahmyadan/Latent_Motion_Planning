#!/bin/bash

eval "$(conda shell.bash hook)"

conda create -n z_plan python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate z_plan 

pip install poetry

pip install -U --force-reinstall charset-normalizer chardet

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

poetry install 

cd latent_motion_planning/envs/HighwayEnv  && git checkout master || exit

pip install -e . 

