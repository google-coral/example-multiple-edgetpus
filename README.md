#Multiple Edge TPU Demo Example(C / C++)

This example shows how build and run a program that will run inferences on a
video with multiple edge tpus.

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
run `bazel build src:hello-world`.

Once bazel builds the project successfully, run the program 
`./bazel-bin/src/hello-world <absolute URI to video file>`.

To clean the project, run `bazel clean`.