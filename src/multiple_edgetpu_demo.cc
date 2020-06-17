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
#include <string>

// To run this file
//    1) Build the file with: bazel build src:multiple_edgetpu_demo
//    2) Run: path/to/executable path/to/video
// NOTE: you must enter the absolute URI for a local video file

int main(int argc, char *argv[]) {
  const int kNumArgs = 2;

  if (argc != kNumArgs) {
    g_print("Usage: <path to gst binary> <video uri>\n");
    return -1;
  }

  // Creating video path
  const std::string user_input = argv[1];
  const std::string video_path = "playbin uri=" + user_input;

  // Initialize GStreamer
  gst_init(NULL, NULL);

  // Build the pipeline
  GstElement *pipeline = gst_parse_launch(video_path.c_str(), NULL);

  // Start playing
  GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);

  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Unable to set the pipeline to the playing state.\n");
    gst_object_unref(pipeline);
    return -1;
  }

  // Wait until error or EOS
  GstBus *bus = gst_element_get_bus(pipeline);
  GstMessage *msg =
      gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS);

  // Free resources
  if (msg != NULL) gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
