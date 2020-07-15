# Multiple Edge TPU Demo Example(C / C++)

This example shows how build and run a program that will run inferences on a
video with multiple Edge TPUs.

## Requirements

This repository contains code that will run with Coral devices:

* [Dev Board](https://coral.ai/products/dev-board/)
* [USB Accelerator (2)](https://coral.ai/products/accelerator/)

You will also need to install [GStreamer](https://gstreamer.freedesktop.org/documentation/installing/) 
and [Bazel](https://docs.bazel.build/versions/master/install.html) to run the projects.

## Native C/C++ Code

All of the C/C++ code are in `src` folder.

## Docker Set-Up

You will need to install docker on your computer first to run the project with
docker or docker-shell.

To run and compile the project inside docker, launch docker-shell by entering
`make DOCKER_IMAGE=debian:buster DOCKER_CPUS="aarch64" docker-shell`. Once the
docker-shell has been setup, enter `make CPU=aarch64 examples` to compile and build
the project. Open a new terminal when the project is finished building, the binary
should be in `out/aarch64/examples/` directory. In order to run the project, you
must be connected to a Coral Dev Board and copy the binary over. All necessary
tflite model files should also be copied to the Coral Dev Board.
