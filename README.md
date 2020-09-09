# Multiple Edge TPU Demo

This demo shows how to use
[model pipelining](https://coral.ai/docs/edgetpu/pipeline/) (with multiple Edge
TPUs) to process a video with a larger model.

## Requirements

This demo assumes the following setup

*   [Dev Board](https://coral.ai/products/dev-board/)
*   [1 or 2 USB Accelerators](https://coral.ai/products/accelerator/)
*   USB Accelerators should be connected to Dev Board with a powered USB hub to
    guarantee sufficient power supply

## Compiling the demo

You will need to install docker on your computer first to compile the project.

```
make DOCKER_TARGETS=demo docker-build
```

The binary should be in `out/aarch64/demo` directory.

## Running the demo

To run the demo, copy `test_data` and generated binary `multiple_edgetpu_demo`
to Dev Board. Run:

```
./multiple_edgetpu_demo --num_segments 3 \
  --video_path <your video>
```

You can enter `./multiple_edgetpu_demo --help` to see all the flag options and
default values.

## Performance expectation

To analyze a 720p 60 FPS video with `inception_v3` model, the system can process
about 20 FPS with 1 Edge TPU; and 50+ FPS with 3 Edge TPUs using model
pipelining.
