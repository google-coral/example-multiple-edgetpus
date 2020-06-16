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

## Linux Instructions

To build a basic project, navigate to the `multiple-edgetpu-demo` folder and 
run `bazel build src:multiple_edgetpu_demo`.

Once bazel builds the project successfully, run the program 
`./bazel-bin/src/multiple_edgetpu_demo <absolute URI to video file>`.

To clean the project, run `bazel clean`.

## Docker Set-Up

You will need to install docker on your computer first to run the project with
docker or docker-shell.

To run and compile the project inside docker, launch docker-shell by entering
`make DOCKER_IMAGE=debian:buster DOCKER_CPUS="aarch64" docker-shell`. Once the
docker-shell has been setup, enter `make CPU=k8 examples` to compile and build
the project. Open a new terminal when the project is finished building, and
navigate to the `multiple_edgetpu_demo` folder and run `out/k8/examples/binary`
and the appropriate command line arguments for the project. 
