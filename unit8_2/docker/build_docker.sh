#!/bin/bash
PARENT=nvidia/cuda:11.8.0-devel-ubuntu20.04
TAG=hf_deeprl/unit8_2
VERSION=1.0

DOCKER_BUILDKIT=1 docker build --build-arg PARENT_IMAGE=${PARENT} . -f Dockerfile -t ${TAG}:${VERSION}