# Unit 8: Proximal Policy Gradient (PPO) with PyTorch

In this unit, we'll use PPO from cleanrl to train a AI on the gym environement
LunarLander-v2.

## code from the collab notebook

Below is the code that was used in the notebook:

```
!apt install python-opengl
!apt install ffmpeg
!apt install xvfb
!pip install pyglet==1.5
!pip3 install pyvirtualdisplay
```

```
!pip install gym==0.21
!pip install imageio-ffmpeg
!pip install huggingface_hub
!pip install box2d
```

Since box2d requires `swig` to be installed, we include through `apt` in the docker file.

## running the code

We take the implementation from cleanrl.


## docker

We will try to reproduce the environement proposed in the google collab notebook.
This means ubuntu 20.04, python 3.8 and so on. See dockerfile.

## running

```
./docker/run_docker_gpu.sh python3 -m src --env-id="LunarLander-v2" --repo-id="toinsson/ppo-cartpole-v0" --total-timesteps=50000
```