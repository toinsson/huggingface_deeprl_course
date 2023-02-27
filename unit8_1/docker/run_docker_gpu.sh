#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

TAG=hf_deeprl/unit8
VERSION=1.0

# fix for gpu
docker run -it --rm --gpus all --privileged\
  --mount src=$(pwd),target=/root/code,type=bind ${TAG}:${VERSION}\
  bash -c "cd /root/code/ && $cmd_line"
