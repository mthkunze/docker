# Fast.ai Docker files


## Overview

This container pulls the latest fast.ai class. You can find this repo here: https://github.com/fastai/fastai and you can learn more about the Fast.ai course here: http://course.fast.ai/

## Requirements:

[Docker CE](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

[NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker)

Nvidia Drivers


## Build

`sudo docker build -t paperspace/fastai .`

## Pre-built runtimes

You can also just run the following without having to build the entire container yourself. This will pull the container from Docker Hub.

`sudo docker run --runtime=nvidia -d -p 8888:8888 paperspace/fastai:cuda9_pytorch0.3.0`