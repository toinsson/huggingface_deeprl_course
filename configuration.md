# Conda

We could use conda as much as possible.

Install for example these packages:
- python=3.10
- jupyter
- notebook
- torch

However, most of the packages rely on `pip`. So we will move instead to docker.


# Nvidia Docker

With popOS, we need to do a bit of hacking to get nvidia-docker installed.
See: https://gist.github.com/kuang-da/2796a792ced96deaf466fdfb7651aa2e

This can then be tested with:
`sudo docker run --rm --gpus all --privileged -v /dev:/dev nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`

We can then modify the docker command from `rl-baselines3-zoo` to use the gpu as well.
`sudo docker run --gpus all --privileged stablebaselines/rl-baselines3-zoo:latest nvidia-smi`

Then we can monitor what is happening from tensorboard:
https://leimao.github.io/blog/TensorBoard-On-Docker/

# Docker HGF

And we can push our model to HF:
sudo ./scripts/run_docker_gpu.sh python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga toinsson -f logs

For this command to work, we had to add git-lfs to the docker image:
`
RUN apt-get -y update \
    && apt-get -y install \
    ...
    sudo \
    wget \
    git \
    ...
`

`
RUN wget https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-linux-amd64-v3.3.0.tar.gz \
    && pwd && ls \
    && tar -xf git-lfs-linux-amd64-v3.3.0.tar.gz \
    && cd git-lfs-3.3.0 \
    && chmod 700 install.sh && ./install.sh
`

and modify the login procedure by using huggingface_hub.login in the python code:
`
from huggingface_hub import login
login("TOKEN")
`

# Additional training

We can continue the training from our agent by using the --trained-agent option.

`./scripts/run_docker_gpu.sh python train.py --algo dqn --env SpaceInvadersNoFrameskip-v4 --n-timesteps 5000000 --tensorboard-log ./logs/tensorboard -i logs/dqn/SpaceInvadersNoFrameskip-v4_14/best_model.zip`


# push to HF hub

`./scripts/run_docker_gpu.sh python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga toinsson -f logs`


# Activate env in docker
https://pythonspeed.com/articles/activate-virtualenv-dockerfile/


https://towardsdatascience.com/training-rl-agents-in-stable-baselines3-is-easy-9d01be04c9db
