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
//    2) Run: path/to/executable path/to/video path/to/model_file

#include <gst/gst.h>
#include <iostream>
#include <string>
#include <vector>
#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"  
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Function that setsup and builds the interpreter for interencing
// Inputs: FlatBufferModel and EdgeTpuContext
// Output: unique_ptr to tflite Interpreter
std::unique_ptr<tflite::Interpreter> SetUpIntepreter(
    const tflite::FlatBufferModel &model, edgetpu::EdgeTpuContext *context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // Register custom op for edge tpu
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  // Build an interpreter with given model
  tflite::InterpreterBuilder builder(model.GetModel(), resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) == kTfLiteOk) {
    std::cout << "Successfully built interpreter!\n";
  } else {
   std::cout << "Error: Cannot build interpreter\n";
  }
  // Add edge tpu as external context to interpreter
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  interpreter->AllocateTensors();
  return interpreter;
}

// Function called on every new available sample, reads each frame data, and
// runs an inference on each frame, printing the result id and scores. Needs an
// appsink in the pipeline in order to work.
// Inputs: GstElement and tflite::Interpreter
// Output: GstFlow value (OK or ERROR)
GstFlowReturn OnNewSample(GstElement *sink, tflite::Interpreter *interpreter) {
  GstFlowReturn retval = GST_FLOW_OK;
  // Retrieve sample from sink
  GstSample *sample;
  g_signal_emit_by_name(sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo info;
    // Read buffer from sample
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    // Read buffer into mapinfo
    if (gst_buffer_map(buffer, &info, GST_MAP_READ) == TRUE) {
      // Allocate input tensor and copy mapinfo data over
      uint8_t *input_tensor = interpreter->typed_input_tensor<uint8_t>(0);
      std::memcpy(input_tensor, info.data, info.size);
      // Run inference on input tensor
      if (interpreter->Invoke() != kTfLiteOk) {
        std::cout << "ERROR: Invoke did not work\n";
      } else {
        std::cout << "Success in invoking\n";
      }
      // Retrieve output tensor
      uint8_t *output_tensor = interpreter->typed_output_tensor<uint8_t>(0);
      int output_index = interpreter->outputs()[0];
      TfLiteTensor *out_tensor = interpreter->tensor(output_index);
      std::vector<float> inference_result(out_tensor->bytes);
      // Dequantize output tensor
      for (unsigned int i = 0; i < inference_result.size(); i++) {
        inference_result[i] =
            (output_tensor[i] - out_tensor->params.zero_point) *
            out_tensor->params.scale;
      }
      // Print result of output tensor (all labels with a score > 0.1)
      for (unsigned int i = 0; i < inference_result.size(); i++) {
        if (inference_result[i] > 0.1) {
          std::cout << "-----------------------------------\n";
          std::cout << "Label id " << i << std::endl;
          std::cout << "Score: " << inference_result[i] << std::endl;
        }
      }
      std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    } else {
      std::cout << "Error: Couldn't get buffer info\n";
      retval = GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &info);
    gst_sample_unref(sample);
  }
  return retval;
}

// Function reads messages as they become available from the stream and
// terminate if necessary.
// Inputs: GstBus, GstMessage, and gpointer
// Output: gboolean (only FALSE when error or warning encountered)
gboolean OnBusMessage(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = reinterpret_cast<GMainLoop *>(data);

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("End of stream\n");
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, nullptr);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *error;
      gst_message_parse_warning(msg, &error, nullptr);
      g_printerr("Warning: %s\n", error->message);
      g_error_free(error);
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
    g_print("Usage: <binary path> <video file path> <model file path>\n");
    return -1;
  }

  // Create classification engine
  const std::string model_path = argv[2];
  // Load model to memory
  tflite::StderrReporter error_reporter;
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str(),
                                                      &error_reporter);
  // Setup edge tpu context
  auto all_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  // Open available device
  auto tpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  std::unique_ptr<tflite::Interpreter> interpreter =
      SetUpIntepreter(*(model), tpu_context.get());

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
  gst_bus_add_watch(bus, OnBusMessage, loop);
  gst_object_unref(bus);

  auto *sink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
  g_signal_connect(sink, "new-sample", reinterpret_cast<GCallback>(OnNewSample),
                   interpreter.get());

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
