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

#include <gst/gst.h>
#include <iostream>
#include <string>

// To run this file
//    1) Build the file with: bazel build src:multiple_edgetpu_demo
//    2) Run: path/to/executable path/to/video

// Function called on every new available sample and reads each frame data.
// Needs an appsink in the pipeline in order to work.
// Input: GstElement
// Output: GstFlow value (OK or ERROR)
GstFlowReturn OnNewSample(GstElement *sink) {
  GstFlowReturn retval = GST_FLOW_OK;
  // getting a sample from the sink
  GstSample *sample;
  g_signal_emit_by_name(sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo info;
    // getting the buffer from the sample
    GstBuffer *buffer = gst_sample_get_buffer(sample);

    // reading buffer into mapinfo
    if (gst_buffer_map(buffer, &info, GST_MAP_READ) == TRUE) {
      std::cout << "size of frame map: " << info.size << std::endl;
      // TODO: run inference on frame data (info.data)
    } else {
      g_printerr("Error: Couldn't get buffer info\n");
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
  const int kNumArgs = 2;

  if (argc != kNumArgs) {
    g_print("Usage: <binary path> <video file path>\n");
    return -1;
  }

  // Creating pipeline
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
  if (sink) {
    printf("Sink exists\n");
  }
  g_signal_connect(sink, "new-sample", reinterpret_cast<GCallback>(OnNewSample),
                   nullptr);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
