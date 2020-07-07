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
//    2) Run: path/to/executable path/to/video path/to/model_file_base
// NOTE: the default number of segments needed is 3

#include <gst/gst.h>
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

// Struct for tflite intepreter and model pipeline runner
struct Container {
  coral::PipelinedModelRunner *runner;
  tflite::Interpreter *interpreter;
};

// Struct for model pipeline runner and gloop
struct LoopContainer {
  GMainLoop *loop;
  coral::PipelinedModelRunner *runner;
};

static std::chrono::time_point<std::chrono::system_clock> start, end;

// Function that returns a vector of Edge TPU contexts
// Returns an empty vector if not enough Edge TPUs are connected
// Input: int num (number of expected Edge TPUs)
// Ouput: vector of shared ptrs to Edge TPU contexts
std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> PrepareEdgeTPUContexts(
    int num) {
  // Get all available Edge TPUs
  const auto &all_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();

  // Open each Edge TPU device and add to vector of Edge TPU contexts
  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts(num);
  for (size_t i = 0; i < all_tpus.size(); i++) {
    edgetpu_contexts[i] =
        CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            all_tpus[i].type, all_tpus[i].path));
    std::cout << "Device " << all_tpus[i].path << " is selected.\n";
  }
  CHECK_EQ(edgetpu_contexts.size(), num);
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
GstFlowReturn OnNewSample(GstElement *sink, Container *cc) {
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
          CHECK_NOTNULL(cc->runner->GetInputTensorAllocator());
      std::vector<coral::PipelineTensor> image_input_tensor;
      // Create pipeline tensor
      coral::PipelineTensor pipe_buffer;
      pipe_buffer.data.data = allocator->alloc(info.size);
      pipe_buffer.bytes = info.size;
      pipe_buffer.type = kTfLiteUInt8;
      memcpy(pipe_buffer.data.data, info.data, info.size);
      image_input_tensor.push_back(pipe_buffer);
      // Push pipeline tensor to Pipeline Runner
      start = std::chrono::system_clock::now();
      CHECK_EQ(cc->runner->Push(image_input_tensor), true);
    } else {
      std::cout << "Error: Couldn't get buffer info" << std::endl;
      retval = GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &info);
    gst_sample_unref(sample);
  }
  return retval;
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
  const int kNumArgs = 3;

  if (argc != kNumArgs) {
    std::cout
        << "Usage: <binary path> <video file path> <model file path segments>"
        << std::endl;
    return -1;
  }

  // Get available devices
  auto tpu_contexts = PrepareEdgeTPUContexts(3);

  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  // Create all model path strings
  const std::string model_path_base = argv[2];
  int num_segments = 2;
  std::vector<std::string> model_path_segments(num_segments);
  for (int i = 0; i < num_segments; i++) {
    model_path_segments[i] = model_path_base + "_segment_" + std::to_string(i) +
                             "_of_" + std::to_string(num_segments) +
                             "_edgetpu.tflite";
  }

  // Create vector of tflite interpreters for model segments
  std::vector<std::unique_ptr<tflite::Interpreter>> managed_interpreters(
      num_segments);
  std::vector<tflite::Interpreter *> all_interpreters(num_segments);
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models(num_segments);
  for (int i = 0; i < num_segments; i++) {
    models[i] = CHECK_NOTNULL(
        tflite::FlatBufferModel::BuildFromFile(model_path_segments[i].c_str()));
    managed_interpreters[i] =
        CHECK_NOTNULL(SetUpIntepreter(*(models[i]), tpu_contexts[i + 1].get()));
    all_interpreters[i] = managed_interpreters[i].get();
  }

  // Create Model Pipeline Runner
  std::unique_ptr<coral::PipelinedModelRunner> runner(
      new coral::PipelinedModelRunner(all_interpreters));
  CHECK_NOTNULL(runner);

  // Create tflite interpreter for whole model
  std::string model_path = model_path_base + "_edgetpu.tflite";
  auto model =
      CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(model_path.c_str()));
  std::unique_ptr<tflite::Interpreter> interpreter =
      CHECK_NOTNULL(SetUpIntepreter(*(model), tpu_contexts[0].get()));
  Container runner_container;
  runner_container.runner = runner.get();
  runner_container.interpreter = interpreter.get();

  // Consumer function prints out Pipeline Runner results as they are pushed
  auto request_consumer = [&runner_container]() {
    std::vector<coral::PipelineTensor> output_tensors;
    static int frame_count = 0;
    while (runner_container.runner->Pop(&output_tensors)) {
      if (output_tensors.size() > 0) {
        coral::PipelineTensor out_tensor = output_tensors[0];
        uint8_t *output_tensor = static_cast<uint8_t *>(out_tensor.data.data);

        // Dequantize output tensor
        const int output_index = runner_container.interpreter->outputs()[0];
        const TfLiteTensor *tensor_data =
            CHECK_NOTNULL(runner_container.interpreter->tensor(output_index));
        std::vector<float> inference_result(out_tensor.bytes);
        for (size_t i = 0; i < inference_result.size(); i++) {
          inference_result[i] =
              (output_tensor[i] - tensor_data->params.zero_point) *
              tensor_data->params.scale;
        }

        // Find and print out max score of each frame inference result
        float max_score = 0.0;
        int max_index = 0;
        for (size_t i = 0; i < inference_result.size(); i++) {
          if (inference_result[i] > max_score) {
            max_score = inference_result[i];
            max_index = i;
          }
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_time = end - start;
        std::cout << "Frame " << frame_count++ << std::endl;
        std::cout << "---------------------------" << std::endl;
        std::cout << "Label id " << max_index << std::endl;
        std::cout << "Max Score: " << max_score << std::endl;
        std::cout << "Inference (sec): " << elapsed_time.count() << std::endl;
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      }
      coral::FreeTensors(output_tensors,
                         runner_container.runner->GetOutputTensorAllocator());
      output_tensors.clear();
    }
  };

  // Create pipeline
  const std::string user_input = argv[1];
  const std::string pipeline_src =
      "filesrc location=" + user_input +
      " ! decodebin ! glfilterbin filter=glbox ! "
      "video/x-raw,width=224,height=224,format=RGB ! "
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
  auto consumer = std::thread(request_consumer);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  consumer.join();

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
