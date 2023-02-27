#!/bin/bash

PARENT=pytorch/pytorch:latest
TAG=hf_deeprl/unit5
VERSION=1.0

docker build --build-arg PARENT_IMAGE=${PARENT} . -f Dockerfile -t ${TAG}:${VERSION}
