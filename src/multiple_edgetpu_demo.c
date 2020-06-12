#include <gst/gst.h>
#include <string.h>

/* To run this file
   1) Build the file with: build bazel path/to/folder:hello-gst
   2) Run: path/to/executable path/to/video
      e.g. ./hello-gst file:///home/user/tmp/video.mp4
      e.g. ./hello-gst https://www.google.com/src/video.webm
   NOTE: you must enter the absolute URI for a local video file
*/

int main(int argc, char *argv[]) {
  int kNumArgs = 2;
  GstElement *pipeline;
  GstBus *bus;
  GstMessage *msg;

  if (argc != kNumArgs) {
    g_print(
        "Usage: <path to gst binary> <video uri>\n"
        "Online video (HTTPS): https://www.website.com/path/to/file\n"
        "Local video: file:///home/username/path/to/file\n");
    return -1;
  }

  /* Creating video path */
  char video_path[] = "playbin uri=";
  char* user_input = argv[1];
  strcat(video_path, user_input);

  /* Initialize GStreamer */
  gst_init(&argc, &argv);

  /* Build the pipeline */
  pipeline = gst_parse_launch(video_path, NULL);

  /* Start playing */
  GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);

  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Unable to set the pipeline to the playing state.\n");
    gst_object_unref(pipeline);
    return -1;
  }

  /* Wait until error or EOS */
  bus = gst_element_get_bus(pipeline);
  msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                   GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

  /* Free resources */
  if (msg != NULL) gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
