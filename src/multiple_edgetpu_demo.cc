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
//    2) Run: executable path/to/video
// Flag options:
// -model_path to specify where the models are located
// -video_path to specify where the video file is located
// -runner_type to specify initial run type
// -num_segments to specify current number of Edge TPUs

#include <gst/gst.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "edgetpu.h"
#include "glog/logging.h"
#include "src/cpp/pipeline/pipelined_model_runner.h"
#include "src/cpp/pipeline/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

ABSL_FLAG(std::string, model_path, "./test/inception_v3_299_quant",
          "Path to the tflite model base.");

ABSL_FLAG(std::string, video_path, "./test/video_device.mp4",
          "Path to the video file.");

ABSL_FLAG(std::string, runner_type, "Runner",
          "Type of runner (Runner or Interpreter).");

ABSL_FLAG(int, num_segments, 3, "Number of segments (Edge TPUs).");

ABSL_FLAG(int, fps, 60, "Framerate of video.");

// Protects: frame_count and run_type in Container struct
absl::Mutex mu_;

// Enumerator for type of interpreter used
enum class RunType {
  kRunner,
  kInterpreter,
};

// Struct for model pipeline runner, tflite::Interpreter, and relevant variables
struct Container {
  coral::PipelinedModelRunner *runner;
  tflite::Interpreter *interpreter;
  float scale;
  int32_t zero_point;
  int frame_count GUARDED_BY(mu_);
  RunType run_type GUARDED_BY(mu_);
};

// Struct for model pipeline runner and gloop
struct LoopContainer {
  GMainLoop *loop;
  coral::PipelinedModelRunner *runner;
  bool loop_finished;
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
  for (int i = 0; i < num; i++) {
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
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  return interpreter;
}

void PrintInferenceResults(Container *container, const uint8_t *data,
                           int size) {
  std::vector<float> inference_result(size);
  // Dequantize output tensor
  for (size_t i = 0; i < inference_result.size(); i++) {
    inference_result[i] = (data[i] - container->zero_point) * container->scale;
  }
  // Find and print out max score of each frame inference result
  auto max_element =
      std::max_element(inference_result.begin(), inference_result.end());
  int max_index = std::distance(inference_result.begin(), max_element);
  float max_score = inference_result[max_index];
  std::cout << "Frame " << container->frame_count++ << std::endl;
  std::cout << "-----------------------------" << std::endl;
  std::cout << "Label id " << max_index << std::endl;
  std::cout << "Max Score: " << max_score << std::endl;
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
      if (container->run_type == RunType::kRunner) {
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
        uint8_t *input_tensor = CHECK_NOTNULL(
            container->interpreter->typed_input_tensor<uint8_t>(0));
        std::memcpy(input_tensor, info.data, info.size);
        // Run inference on input tensor
        const auto &start = std::chrono::system_clock::now();
        CHECK_EQ(container->interpreter->Invoke(), kTfLiteOk);
        // Retrieve output tensor and output tensor data
        const int output_index = container->interpreter->outputs()[0];
        const TfLiteTensor *out_tensor =
            CHECK_NOTNULL(container->interpreter->tensor(output_index));
        const uint8_t *out_tensor_data =
            tflite::GetTensorData<uint8_t>(out_tensor);
        mu_.Lock();
        PrintInferenceResults(container, out_tensor_data, out_tensor->bytes);
        mu_.Unlock();
        std::chrono::duration<double, std::milli> elapsed_time =
            std::chrono::system_clock::now() - start;
        std::cout << "Interpreter: " << elapsed_time.count() << " ms"
                  << std::endl;
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      }
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
  std::chrono::time_point<std::chrono::system_clock> start;
  // Only used for the the first frame inference
  start = std::chrono::system_clock::now();
  while (runner_container->runner->Pop(&output_tensors)) {
    mu_.Lock();
    coral::PipelineTensor out_tensor = output_tensors[0];
    const uint8_t *output_tensor = static_cast<uint8_t *>(out_tensor.data.data);
    PrintInferenceResults(runner_container, output_tensor, out_tensor.bytes);
    std::chrono::duration<double, std::milli> elapsed_time =
        std::chrono::system_clock::now() - start;
    start = std::chrono::system_clock::now();
    std::cout << "Pipeline: " << elapsed_time.count() << " ms" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    coral::FreeTensors(output_tensors,
                       runner_container->runner->GetOutputTensorAllocator());
    mu_.Unlock();
    output_tensors.clear();
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
      container->loop_finished = true;
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, nullptr);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      container->runner->Push({});
      container->loop_finished = true;
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *error;
      gst_message_parse_warning(msg, &error, nullptr);
      g_printerr("Warning: %s\n", error->message);
      g_error_free(error);
      container->runner->Push({});
      container->loop_finished = true;
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  // Create all model path strings
  const std::string model_path_base = absl::GetFlag(FLAGS_model_path);
  const int num_segments = absl::GetFlag(FLAGS_num_segments);

  // Get available devices
  auto tpu_contexts = PrepareEdgeTPUContexts(num_segments);
  // Create model segment file path strings
  std::vector<std::string> model_path_segments(num_segments);
  for (int i = 0; i < num_segments; i++) {
    model_path_segments[i] = absl::Substitute(
        "$0_segment_$1_of_$2_edgetpu.tflite", model_path_base, i, num_segments);
    std::cout << model_path_segments[i] << std::endl;
  }
  const std::string full_model_path =
      absl::StrCat(model_path_base, "_edgetpu.tflite");

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
  int width, height;
  int input_num = all_interpreters[0]->inputs()[0];
  TfLiteTensor *input_tensor =
      CHECK_NOTNULL(all_interpreters[0]->tensor(input_num));
  width = input_tensor->dims->data[1];
  height = input_tensor->dims->data[2];

  // Get scale and zero_point variables from output tensor
  const int output_index = all_interpreters[num_segments - 1]->outputs()[0];
  const TfLiteTensor *tensor_data =
      CHECK_NOTNULL(all_interpreters[num_segments - 1]->tensor(output_index));
  runner_container.scale = tensor_data->params.scale;
  runner_container.zero_point = tensor_data->params.zero_point;

  // Create tflite::Interpreter
  auto full_model = CHECK_NOTNULL(
      tflite::FlatBufferModel::BuildFromFile(full_model_path.c_str()));
  std::unique_ptr<tflite::Interpreter> interpreter = CHECK_NOTNULL(
      SetUpIntepreter(*(full_model), tpu_contexts[num_segments - 1].get()));

  // Create Model Pipeline Runner
  std::unique_ptr<coral::PipelinedModelRunner> runner(
      new coral::PipelinedModelRunner(all_interpreters));
  CHECK_NOTNULL(runner);
  runner_container.runner = runner.get();
  runner_container.interpreter = interpreter.get();
  runner_container.frame_count = 0;

  // Determine initial run type of program
  const std::string run_type = absl::GetFlag(FLAGS_runner_type);
  if (run_type == "Runner") {
    runner_container.run_type = RunType::kRunner;
  } else {
    runner_container.run_type = RunType::kInterpreter;
  }

  // Create Gstreamer pipeline
  const std::string video_path = absl::GetFlag(FLAGS_video_path);
  const int fps = absl::GetFlag(FLAGS_fps);
  const std::string pipeline_src = absl::Substitute(
      "filesrc location=$0 ! decodebin ! glfilterbin filter=glbox ! videorate "
      "! video/x-raw,framerate=$1/1,width=$2,height=$3,format=RGB ! appsink "
      "name=appsink emit-signals=true max-buffers=1 drop=true",
      video_path, fps, width, height);

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
  loop_container.loop_finished = false;
  gst_bus_add_watch(bus, OnBusMessage, &loop_container);
  gst_object_unref(bus);

  auto *sink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
  g_signal_connect(sink, "new-sample", reinterpret_cast<GCallback>(OnNewSample),
                   &runner_container);

  auto keyboard_watch = [&runner_container, &loop_container]() {
    char input;
    while (!loop_container.loop_finished) {
      std::cin >> input;
      if (input == 'n') {
        mu_.Lock();
        if (runner_container.run_type == RunType::kRunner) {
          runner_container.run_type = RunType::kInterpreter;
          std::cout << "Switching to interpreter" << std::endl;
        } else {
          runner_container.run_type = RunType::kRunner;
          std::cout << "Switching to runner" << std::endl;
        }
        mu_.Unlock();
      }
    }
  };

  // Start consumer and keyboard watching thread
  auto consumer = std::thread(ResultConsumer, &runner_container);
  auto watcher = std::thread(keyboard_watch);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  consumer.join();
  watcher.join();

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
