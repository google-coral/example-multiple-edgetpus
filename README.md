# Multiple Edge TPU Demo

This demo shows how to use
[model pipelining](https://coral.ai/docs/edgetpu/pipeline/) (with multiple Edge
TPUs) to process a video with a larger model.

This demo allows two configurations, one for the [Dev Board](https://coral.ai/products/dev-board/)
and one for generic platforms (including x86_64, ARM64, and ARMv7). The Dev
Board configuration is optimized to run efficiently for the platform. The
generic version relies on SW rendering and decoding to ensure compatability.

## Requirements
### Dev Board Requirements

*   [Dev Board](https://coral.ai/products/dev-board/)
*   [1 or 2 USB Accelerators](https://coral.ai/products/accelerator/)
*   USB Accelerators should be connected to Dev Board with a powered USB hub to
    guarantee sufficient power supply

For the Dev Board, This demo is tested on [Mendel Eagle](https://coral.ai/software/) with
[Eel2 release](https://github.com/google-coral/edgetpu/commit/c48c88871fd3d2e10d298126cd6a08b88d22496c)

### Generic Platforms

*   A x86_64, ARM64, ARMv7 host with multiple EdgeTPUs attached (via USB, PCIe,
or both).

For generic platforms, this is tested on a Linux x86_64 host with USB
accelerators.

## Compiling the demo

You will need to install docker on your computer first to compile the project.

In order to build all CPU targets type:

```
make DOCKER_TARGETS=demo docker-build
```

You can also specify the CPU targets you'd like:
```
make DOCKER_TARGETS=demo DOCKER_CPUS="k8 aarch64" docker-build
```

The binary should be in `out/$(ARCH)/demo` directory.

## Running the demo

To run the demo, copy `test_data` and generated binary `multiple_edgetpu_demo`
to your device. Run:

```
./multiple_edgetpu_demo --num_segments 3 \
  --video_path <your video>
```

You can enter `./multiple_edgetpu_demo --help` to see all the flag options and
default values.

## Performance expectation

To analyze a 720p 60 FPS video with `inception_v3` model, the system can process
about 20 FPS with 1 Edge TPU; and 50+ FPS with 3 Edge TPUs using model
pipelining. Performance may vary in the generic configuration based on SW
decoding and rendering ability.

## License

[Apache License 2.0](LICENSE)

