#!/bin/bash

PARENT=pytorch/pytorch:latest
TAG=hf_deeprl/unit4
VERSION=3.0

docker build --build-arg PARENT_IMAGE=${PARENT} . -f Dockerfile -t ${TAG}:${VERSION}
