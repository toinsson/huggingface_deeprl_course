#!/bin/bash

# PARENT=pytorch/pytorch:latest
# PARENT=nvidia/cuda:11.7.0-devel-ubuntu22.04
PARENT=nvidia/cuda:11.8.0-devel-ubuntu20.04
TAG=hf_deeprl/unit8
VERSION=1.0

docker build --build-arg PARENT_IMAGE=${PARENT} . -f Dockerfile -t ${TAG}:${VERSION}
