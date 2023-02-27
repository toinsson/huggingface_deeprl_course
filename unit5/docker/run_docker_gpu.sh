#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

TAG=hf_deeprl/unit4
VERSION=3.0

# original
# docker run -it --runtime=nvidia --rm --network host --ipc=host \
#   --mount src=$(pwd),target=/root/code/rl_zoo3,type=bind stablebaselines/rl-baselines3-zoo:latest\
#   bash -c "cd /root/code/rl_zoo3/ && $cmd_line"

# fix for xvfb
# docker run -it --init --runtime=nvidia --rm --ipc=host \
#   --mount src=$(pwd),target=/root/code/rl_zoo3,type=bind stablebaselines/rl-baselines3-zoo:latest\
#   bash -c "cd /root/code/rl_zoo3/ && $cmd_line"

# fix for gpu
docker run -it --rm --gpus all --privileged\
  --mount src=$(pwd),target=/root/code,type=bind ${TAG}:${VERSION}\
  bash -c "cd /root/code/ && $cmd_line"
