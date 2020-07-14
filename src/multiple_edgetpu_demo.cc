// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// To run this file
//    1) Build the file with: make CPU=aarch64 examples
//    2) Run: executable path/to/video path/to/model_file_base
//    num_segments
// NOTE: the default number of segments needed is 3

#include <gst/gst.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "edgetpu.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Struct for model pipeline runner and relevant variables
struct Container {
  coral::PipelinedModelRunner *runner;
  float scale;
  int32_t zero_point;
};

// Struct for model pipeline runner and gloop
struct LoopContainer {
  GMainLoop *loop;
  coral::PipelinedModelRunner *runner;
};

// Function that returns a vector of Edge TPU contexts
// Only returns a vector of Edge TPU contexts if enough are available
// Otherwise, program will terminate
// Input: int num (number of expected Edge TPUs)
// Ouput: vector of shared ptrs to Edge TPU contexts
std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> PrepareEdgeTPUContexts(
    int num) {
  // Get all available Edge TPUs
  const auto &all_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  CHECK_GE(all_tpus.size() >= num, true);

  // Open each Edge TPU device and add to vector of Edge TPU contexts
  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts(num);
  for (size_t i = 0; i < num; i++) {
    edgetpu_contexts[i] =
        CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            all_tpus[i].type, all_tpus[i].path));
    std::cout << "Device " << all_tpus[i].path << " is selected." << std::endl;
  }
  return edgetpu_contexts;
}

// Function that sets up and builds the interpreter for inference
// Inputs: FlatBufferModel and EdgeTpuContext
// Output: unique_ptr to tflite Interpreter
std::unique_ptr<tflite::Interpreter> SetUpIntepreter(
    const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // Register custom op for edge TPU
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  // Build an interpreter with given model
  tflite::InterpreterBuilder builder(model.GetModel(), resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  TfLiteStatus interpreter_build = builder(&interpreter);
  CHECK_EQ(interpreter_build, kTfLiteOk);
  CHECK_EQ(interpreter->tensor(interpreter->inputs()[0])->type, kTfLiteUInt8);
  CHECK_EQ(interpreter->tensor(interpreter->outputs()[0])->type, kTfLiteUInt8);

  // Add edge TPU as external context to interpreter
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  interpreter->AllocateTensors();
  return interpreter;
}

// Function called on every new available sample, reads each frame data, and
// runs an inference on each frame, printing the result id and scores. Needs an
// appsink in the pipeline in order to work.
// Inputs: GstElement* and Container*
// Output: GstFlow value (OK or ERROR)
GstFlowReturn OnNewSample(GstElement *sink, Container *container) {
  GstFlowReturn retval = GST_FLOW_OK;
  // Retrieve sample from sink
  GstSample *sample;
  g_signal_emit_by_name(sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo info;
    // Read buffer from sample
    GstBuffer *buffer = CHECK_NOTNULL(gst_sample_get_buffer(sample));
    // Read buffer into mapinfo
    if (gst_buffer_map(buffer, &info, GST_MAP_READ) == TRUE) {
      coral::Allocator *allocator =
          CHECK_NOTNULL(container->runner->GetInputTensorAllocator());
      // Create pipeline tensor
      coral::PipelineTensor pipe_buffer;
      pipe_buffer.data.data = allocator->alloc(info.size);
      pipe_buffer.bytes = info.size;
      pipe_buffer.type = kTfLiteUInt8;
      std::memcpy(pipe_buffer.data.data, info.data, info.size);
      // Push pipeline tensor to Pipeline Runner
      CHECK_EQ(container->runner->Push({pipe_buffer}), true);
    } else {
      std::cout << "Error: Couldn't get buffer info" << std::endl;
      retval = GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &info);
    gst_sample_unref(sample);
  }
  return retval;
}

// Function that waits for outpute Pipeline Tensors and prints inference results
// from Model Pipeline Runner
// Input: Container*
void ResultConsumer(Container *runner_container) {
  std::vector<coral::PipelineTensor> output_tensors;
  static int frame_count;
  std::chrono::time_point<std::chrono::system_clock> start;
  // Only used for the the first frame inference
  start = std::chrono::system_clock::now();
  while (runner_container->runner->Pop(&output_tensors)) {
    coral::PipelineTensor out_tensor = output_tensors[0];
    uint8_t *output_tensor = static_cast<uint8_t *>(out_tensor.data.data);

    // Dequantize output tensor
    std::vector<float> inference_result(out_tensor.bytes);
    for (size_t i = 0; i < inference_result.size(); i++) {
      inference_result[i] = (output_tensor[i] - runner_container->zero_point) *
                            runner_container->scale;
    }

    // Find and print out max score of each frame inference result
    auto max_element =
        std::max_element(inference_result.begin(), inference_result.end());
    int max_index = std::distance(inference_result.begin(), max_element);
    float max_score = inference_result[max_index];
    std::chrono::duration<double, std::milli> elapsed_time =
        std::chrono::system_clock::now() - start;
    std::cout << "Frame " << frame_count++ << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Label id " << max_index << std::endl;
    std::cout << "Max Score: " << max_score << std::endl;
    std::cout << "Runner: " << elapsed_time.count() << " ms" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    coral::FreeTensors(output_tensors,
                       runner_container->runner->GetOutputTensorAllocator());
    output_tensors.clear();
    start = std::chrono::system_clock::now();
  }
}

// Function reads messages as they become available from the stream and
// terminate if necessary.
// Inputs: GstBus, GstMessage, and gpointer (to LoopContainer)
// Output: gboolean (only FALSE when error or warning encountered)
gboolean OnBusMessage(GstBus *bus, GstMessage *msg, gpointer data) {
  LoopContainer *container = reinterpret_cast<LoopContainer *>(data);
  GMainLoop *loop = container->loop;

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("End of stream\n");
      container->runner->Push({});
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, nullptr);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      container->runner->Push({});
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *error;
      gst_message_parse_warning(msg, &error, nullptr);
      g_printerr("Warning: %s\n", error->message);
      g_error_free(error);
      container->runner->Push({});
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int main(int argc, char *argv[]) {
  const int kNumArgs = 4;

  if (argc != kNumArgs) {
    std::cout << "Usage: <binary path> <video file path> <model file path "
                 "segments> <number of segments>"
              << std::endl;
    return -1;
  }

  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  // Create all model path strings
  const std::string model_path_base = argv[2];
  int num_segments = std::stoi(argv[3]);

  // Get available devices
  auto tpu_contexts = PrepareEdgeTPUContexts(num_segments);
  // Create model segment file path strings
  std::vector<std::string> model_path_segments(num_segments);
  for (int i = 0; i < num_segments; i++) {
    model_path_segments[i] = model_path_base + "_segment_" + std::to_string(i) +
                             "_of_" + std::to_string(num_segments) +
                             "_edgetpu.tflite";
  }

  Container runner_container;

  // Create vector of tflite interpreters for model segments
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter *> all_interpreters(num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  for (int i = 0; i < num_segments; i++) {
    models[i] = CHECK_NOTNULL(
        tflite::FlatBufferModel::BuildFromFile(model_path_segments[i].c_str()));
    managed_interpreters[i] =
        CHECK_NOTNULL(SetUpIntepreter(*(models[i]), tpu_contexts[i].get()));
    all_interpreters[i] = managed_interpreters[i].get();
  }

  // Get tensor dimensions from input tensor for gstreamer pipeline
  int width, height, depth;
  int input_num = all_interpreters[0]->inputs()[0];
  TfLiteTensor *input_tensor =
      CHECK_NOTNULL(all_interpreters[0]->tensor(input_num));
  width = input_tensor->dims->data[1];
  height = input_tensor->dims->data[2];
  depth = input_tensor->dims->data[3];

  // Get scale and zero_point variables from output tensor
  const int output_index = all_interpreters[num_segments - 1]->outputs()[0];
  const TfLiteTensor *tensor_data =
      CHECK_NOTNULL(all_interpreters[num_segments - 1]->tensor(output_index));
  runner_container.scale = tensor_data->params.scale;
  runner_container.zero_point = tensor_data->params.zero_point;

  // Create Model Pipeline Runner
  std::unique_ptr<coral::PipelinedModelRunner> runner(
      new coral::PipelinedModelRunner(all_interpreters));
  CHECK_NOTNULL(runner);
  runner_container.runner = runner.get();

  // Create pipeline
  const std::string user_input = argv[1];
  const std::string pipeline_src = "filesrc location=" + user_input +
                                   " ! decodebin ! glfilterbin filter=glbox ! "
                                   "video/x-raw,width=" +
                                   std::to_string(width) +
                                   ",height=" + std::to_string(height) +
                                   ",format=RGB ! "
                                   "appsink name=appsink emit-signals=true "
                                   "max-buffers=1 drop=true";

  // Initialize GStreamer
  gst_init(NULL, NULL);

  // Build the pipeline
  GstElement *pipeline = gst_parse_launch(pipeline_src.c_str(), NULL);
  auto loop = g_main_loop_new(nullptr, FALSE);

  // Setting up bus watcher
  GstBus *bus = gst_element_get_bus(pipeline);
  LoopContainer loop_container;
  loop_container.loop = loop;
  loop_container.runner = runner.get();
  gst_bus_add_watch(bus, OnBusMessage, &loop_container);
  gst_object_unref(bus);

  auto *sink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
  g_signal_connect(sink, "new-sample", reinterpret_cast<GCallback>(OnNewSample),
                   &runner_container);

  // Start consumer thread
  auto consumer = std::thread(ResultConsumer, &runner_container);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  consumer.join();

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
