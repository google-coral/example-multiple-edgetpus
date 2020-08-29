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

#include <gdk/gdk.h>
#include <gdk/gdkwayland.h>
#include <glib-unix.h>
#include <gst/gl/gl.h>
#include <gst/gst.h>
#include <gtk/gtk.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
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

ABSL_FLAG(std::string, model_path, "./test_data/inception_v3_299_quant",
          "Path to the tflite model base.");

ABSL_FLAG(std::string, video_path, "./test_data/video_device.mp4",
          "Path to the video file.");

ABSL_FLAG(std::string, labels_path, "./test_data/imagenet_labels.txt",
          "Path to labels file.");

ABSL_FLAG(bool, use_multiple_edgetpu, true,
          "Using PipelinedModelRunner or tflite::Interpreter.");

ABSL_FLAG(int, num_segments, 3, "Number of segments (Edge TPUs).");

ABSL_FLAG(bool, visualize, true, "Display video and inference results.");

ABSL_FLAG(bool, loop, true, "Loop video forever.");

const float kFontSizePercent = 0.05;
const char *kSvgHeader = "<svg width=\"$0\" height=\"$1\" version=\"1.1\">\n";
const char *kSvgFooter = "</svg>\n";
const char *kSvgText =
    " <text x=\"$0\" y=\"$1\" dx=\"0.05em\" dy=\"0.05em\" class=\"text "
    "text_bg$3\">$2</text>\n"
    "  <text x=\"$0\" y=\"$1\" class=\"text text_fg$3\">$2</text>\n";
const char *kSvgStyles =
    " <style>\n"
    "  .text { font-size: $0px; font-family: sans-serif; }\n"
    "  .text_bg { fill: black; }\n"
    "  .text_fg { fill: white; }\n"
    "  .text_ra { text-anchor: end; }\n"
    " </style>\n";

class GstWrapperAllocator : public coral::Allocator {
 public:
  GstWrapperAllocator() = default;

  virtual ~GstWrapperAllocator() {
    CHECK(map_infos_.empty());
    CHECK(samples_.empty());
  }

  virtual void *alloc(size_t size) {
    // This is not a real allocator, can't allocate new memory.
    CHECK(false) << "Unsupported operation";
    return nullptr;
  }

  virtual void free(void *p, size_t size) {
    absl::MutexLock lock(&mutex_);

    auto info_iter = map_infos_.find(p);
    auto sample_iter = samples_.find(p);
    CHECK(info_iter != map_infos_.end());
    CHECK(sample_iter != samples_.end());
    map_infos_.erase(info_iter);
    samples_.erase(sample_iter);

    GstSample *sample = sample_iter->second;
    GstMapInfo info = info_iter->second;
    gst_buffer_unmap(CHECK_NOTNULL(gst_sample_get_buffer(sample)), &info);
    gst_sample_unref(sample);
  }

  void *wrap(GstSample *sample) {
    absl::MutexLock lock(&mutex_);

    // TODO: This is using an experimental and unstable API that
    // will change in the future. In particular there will be no
    // need to map buffer for CPU access here.
    GstMapInfo info;
    GstBuffer *buffer = CHECK_NOTNULL(gst_sample_get_buffer(sample));
    CHECK(gst_buffer_map(buffer, &info, GST_MAP_READ));
    void *data = reinterpret_cast<void *>(info.data);
    map_infos_[data] = info;
    samples_[data] = sample;
    return data;
  }

 private:
  absl::Mutex mutex_;
  std::unordered_map<void *, GstMapInfo> map_infos_;
  std::unordered_map<void *, GstSample *> samples_;
};

class App {
 public:
  App();
  ~App();
  void Run();

 private:
  static std::unique_ptr<tflite::Interpreter> InitializeInterpreter(
      tflite::FlatBufferModel *model, edgetpu::EdgeTpuContext *context);

  void InitializeInference();
  void InitializeGstreamer();
  void InitializeGtkWindow();
  void DestroyGtkWindow();
  gboolean OnBusMessage(GstMessage *msg);
  GstBusSyncReply OnBusMessageSync(GstMessage *msg);
  GstFlowReturn OnNewSample(GstElement *element);
  void ConsumeRunner();
  void HandleOutput(const uint8_t *data, double duration_ms);
  void Quit();

  std::unordered_map<int, std::string> labels_;
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models_;
  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> edgetpu_contexts_;
  std::vector<std::unique_ptr<tflite::Interpreter>> segment_interpreters_;
  std::unique_ptr<coral::PipelinedModelRunner> runner_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> start_times_;
  GstElement *pipeline_ = nullptr;
  GstElement *glsink_ = nullptr;
  GtkWidget *window_ = nullptr;
  GtkWidget *drawing_area_ = nullptr;
  GstWrapperAllocator allocator_;
  int input_width_;
  int input_height_;
  size_t input_bytes_;
  float output_scale_;
  int32_t output_zero_point_;
  size_t output_bytes_;
  size_t output_frames_ = 0;
  int video_width_ = 0;
  int video_height_ = 0;
  bool use_runner_;
  bool running_;
  bool loop_;
  absl::Mutex mutex_;
};

App::App() {
  gst_init(NULL, NULL);
  gtk_init(NULL, NULL);
  running_ = false;
}

App::~App() {
  Quit();

  if (glsink_) {
    gst_object_unref(glsink_);
  }

  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }

  DestroyGtkWindow();
}

std::unique_ptr<tflite::Interpreter> App::InitializeInterpreter(
    tflite::FlatBufferModel *model, edgetpu::EdgeTpuContext *context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  tflite::InterpreterBuilder builder(model->GetModel(), resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  CHECK_EQ(builder(&interpreter), kTfLiteOk);
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  CHECK_EQ(interpreter->tensor(interpreter->inputs()[0])->type, kTfLiteUInt8);
  CHECK_EQ(interpreter->tensor(interpreter->outputs()[0])->type, kTfLiteUInt8);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  return interpreter;
}

void App::InitializeInference() {
  const std::string model_path_base = absl::GetFlag(FLAGS_model_path);
  const size_t num_segments = absl::GetFlag(FLAGS_num_segments);
  const std::string labels_path = absl::GetFlag(FLAGS_labels_path);

  const std::string full_model_path =
      absl::StrCat(model_path_base, "_edgetpu.tflite");
  std::cout << "Full model: " << full_model_path << std::endl;

  std::vector<std::string> model_path_segments(num_segments);
  for (size_t i = 0; i < num_segments; ++i) {
    model_path_segments[i] = absl::Substitute(
        "$0_segment_$1_of_$2_edgetpu.tflite", model_path_base, i, num_segments);
    std::cout << "Segment " << i << ": " << model_path_segments[i] << std::endl;
  }

  const auto &all_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  std::cout << "TPUs needed: " << num_segments << std::endl;
  std::cout << "TPUs found: " << all_tpus.size() << std::endl;
  CHECK_GE(all_tpus.size(), num_segments);

  edgetpu_contexts_.resize(num_segments);
  for (size_t i = 0; i < num_segments; ++i) {
    edgetpu_contexts_[i] =
        CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            all_tpus[i].type, all_tpus[i].path));
    std::cout << "Device " << all_tpus[i].path << " opened" << std::endl;
  }

  segment_interpreters_.resize(num_segments);
  std::vector<tflite::Interpreter *> runner_interpreters(num_segments);
  for (size_t i = 0; i < num_segments; ++i) {
    models_.push_back(CHECK_NOTNULL(tflite::FlatBufferModel::BuildFromFile(
        model_path_segments[i].c_str())));
    segment_interpreters_[i] =
        InitializeInterpreter(models_[i].get(), edgetpu_contexts_[i].get());
    runner_interpreters[i] = segment_interpreters_[i].get();
    std::cout << model_path_segments[i] << " loaded" << std::endl;
  }
  runner_ = std::make_unique<coral::PipelinedModelRunner>(runner_interpreters,
                                                          &allocator_);
  CHECK_NOTNULL(runner_);

  models_.push_back(CHECK_NOTNULL(
      tflite::FlatBufferModel::BuildFromFile(full_model_path.c_str())));
  interpreter_ = InitializeInterpreter(models_.back().get(),
                                       edgetpu_contexts_.back().get());
  std::cout << full_model_path << " loaded" << std::endl;

  auto dims = interpreter_->input_tensor(0)->dims;
  CHECK_EQ(dims->size, 4);
  input_width_ = dims->data[2];
  input_height_ = dims->data[1];
  input_bytes_ = dims->data[0] * dims->data[1] * dims->data[2] * dims->data[3];
  std::cout << absl::StrFormat("Input %dx%d, %zu bytes", input_width_,
                               input_height_, input_bytes_)
            << std::endl;

  const TfLiteTensor *output_tensor =
      CHECK_NOTNULL(interpreter_->output_tensor(0));
  output_bytes_ = output_tensor->bytes;
  output_scale_ = output_tensor->params.scale;
  output_zero_point_ = output_tensor->params.zero_point;
  std::cout << absl::StrFormat("Output %zu bytes, scale %f, zero_point %d",
                               output_bytes_, output_scale_, output_zero_point_)
            << std::endl;

  std::ifstream file(labels_path);
  CHECK(file.is_open()) << "Error opening " << labels_path;
  std::string line;
  while (getline(file, line)) {
    absl::RemoveExtraAsciiWhitespace(&line);
    std::vector<std::string> fields =
        absl::StrSplit(line, absl::MaxSplits(' ', 1));
    if (fields.size() == 2) {
      int label_id;
      CHECK(absl::SimpleAtoi(fields[0], &label_id)) << "malformed label file";
      labels_[label_id] = fields[1];
    }
  }
  std::cout << labels_.size() << " labels loaded from " << labels_path
            << std::endl;
}

void App::InitializeGstreamer() {
  const std::string video_path = absl::GetFlag(FLAGS_video_path);
  const std::string overlay_src =
      "filesrc location=$0 ! decodebin ! glupload ! tee name=t\n"
      " t. ! queue ! glsvgoverlaysink name=glsink\n"
      " t. ! queue ! "
      "glfilterbin filter=glbox ! "
      "video/x-raw,width=$1,height=$2,format=RGB\n"
      "    ! queue max-size-buffers=1 leaky=downstream"
      " ! appsink name=appsink emit-signals=true";
  const std::string headless_src =
      "filesrc location=$0 ! decodebin ! glfilterbin filter=glbox\n"
      "  ! video/x-raw,width=$1,height=$2,format=RGB\n"
      "  ! appsink name=appsink emit-signals=true";

  std::string src = absl::Substitute(
      absl::GetFlag(FLAGS_visualize) ? overlay_src : headless_src,
      absl::GetFlag(FLAGS_video_path), input_width_, input_height_);
  std::cout << std::endl << "GStreamer pipeline:" << std::endl;
  std::cout << src << std::endl;

  pipeline_ = CHECK_NOTNULL(gst_parse_launch(src.c_str(), NULL));
  glsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "glsink");

  GstBus *bus = gst_element_get_bus(pipeline_);
  gst_bus_add_watch(
      bus,
      reinterpret_cast<GstBusFunc>(
          +[](GstBus *bus, GstMessage *msg, App *self) -> gboolean {
            return self->OnBusMessage(msg);
          }),
      this);
  gst_bus_set_sync_handler(
      bus,
      reinterpret_cast<GstBusSyncHandler>(
          +[](GstBus *bus, GstMessage *msg, App *self) -> GstBusSyncReply {
            return self->OnBusMessageSync(msg);
          }),
      this, NULL);
  gst_object_unref(bus);

  GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
  CHECK_NOTNULL(appsink);
  g_signal_connect(appsink, "new-sample",
                   reinterpret_cast<GCallback>(
                       +[](GstElement *element, App *self) -> GstFlowReturn {
                         return self->OnNewSample(element);
                       }),
                   this);
  gst_object_unref(appsink);
}

void App::InitializeGtkWindow() {
  window_ = CHECK_NOTNULL(gtk_window_new(GTK_WINDOW_TOPLEVEL));
  drawing_area_ = CHECK_NOTNULL(gtk_drawing_area_new());

  gtk_window_fullscreen(GTK_WINDOW(window_));

  g_signal_connect(
      drawing_area_, "configure-event",
      G_CALLBACK(+[](GtkWidget *widget, GdkEvent *event, App *self) -> void {
        GtkAllocation alloc;
        gtk_widget_get_allocation(widget, &alloc);
        gst_video_overlay_set_render_rectangle(GST_VIDEO_OVERLAY(self->glsink_),
                                               alloc.x, alloc.y, alloc.width,
                                               alloc.height);
      }),
      this);
  g_signal_connect(glsink_, "drawn",
                   G_CALLBACK(+[](GstElement *object, App *self) -> gboolean {
                     gtk_widget_queue_draw(self->drawing_area_);
                     return FALSE;
                   }),
                   this);
  gtk_container_add(GTK_CONTAINER(window_), drawing_area_);
  gtk_widget_realize(drawing_area_);

  // Wayland support only, no X11.
  GdkWindow *gdk_window = gtk_widget_get_window(drawing_area_);
  GdkDisplay *gdk_display = gdk_window_get_display(gdk_window);
  CHECK(GDK_IS_WAYLAND_DISPLAY(gdk_display));

  gst_video_overlay_set_window_handle(
      GST_VIDEO_OVERLAY(glsink_),
      reinterpret_cast<guintptr>(
          gdk_wayland_window_get_wl_surface(gdk_window)));
  struct wl_display *dpy = gdk_wayland_display_get_wl_display(gdk_display);
  GstContext *context =
      gst_context_new("GstWaylandDisplayHandleContextType", TRUE);
  GstStructure *s = gst_context_writable_structure(context);
  gst_structure_set(s, "display", G_TYPE_POINTER, dpy, NULL);
  gst_element_set_context(glsink_, context);
  gtk_widget_show_all(window_);
}

void App::DestroyGtkWindow() {
  if (drawing_area_) {
    gtk_widget_destroy(drawing_area_);
    drawing_area_ = nullptr;
  }

  if (window_) {
    gtk_widget_destroy(window_);
    window_ = nullptr;
  }
}

gboolean App::OnBusMessage(GstMessage *msg) {
  bool push = false;
  bool quit = false;

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS: {
      gboolean seekable = FALSE;
      if (loop_) {
        GstQuery *query = gst_query_new_seeking(GST_FORMAT_TIME);
        if (gst_element_query(pipeline_, query)) {
          gst_query_parse_seeking(query, NULL, &seekable, NULL, NULL);
        }
        gst_query_unref(query);
      }
      if (!(seekable && gst_element_seek_simple(
                            pipeline_, GST_FORMAT_TIME,
                            static_cast<GstSeekFlags>(GST_SEEK_FLAG_FLUSH |
                                                      GST_SEEK_FLAG_KEY_UNIT),
                            0))) {
        std::cout << "End of stream" << std::endl;
        push = quit = true;
      }
      break;
    }
    case GST_MESSAGE_ERROR: {
      GError *error = NULL;
      gchar *dbg = NULL;
      gst_message_parse_error(msg, &error, &dbg);
      std::cout << absl::StrFormat("ERROR from element %s: %s (%s)",
                                   GST_OBJECT_NAME(msg->src), error->message,
                                   dbg ? dbg : "no dbg");
      g_error_free(error);
      g_free(dbg);
      push = quit = true;
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *error;
      gchar *dbg = NULL;
      gst_message_parse_warning(msg, &error, &dbg);
      std::cout << absl::StrFormat("WARNING from element %s: %s (%s)",
                                   GST_OBJECT_NAME(msg->src), error->message,
                                   dbg ? dbg : "no dbg");
      g_error_free(error);
      g_free(dbg);
      break;
    }
    default:
      break;
  }

  if (push) {
    runner_->Push({});
  }

  if (quit) {
    Quit();
  }

  return TRUE;
}

GstBusSyncReply App::OnBusMessageSync(GstMessage *msg) {
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_NEED_CONTEXT && glsink_) {
    const gchar *ctx_type;
    if (gst_message_parse_context_type(msg, &ctx_type) &&
        !strcmp(ctx_type, GST_GL_DISPLAY_CONTEXT_TYPE)) {
      GstElement *vosink =
          gst_bin_get_by_interface(GST_BIN(glsink_), GST_TYPE_VIDEO_OVERLAY);
      if (vosink) {
        GstGLContext *glcontext = NULL;
        g_object_get(vosink, "context", &glcontext, NULL);
        if (glcontext) {
          GstContext *gstcontext =
              gst_context_new(GST_GL_DISPLAY_CONTEXT_TYPE, TRUE);
          gst_context_set_gl_display(gstcontext,
                                     gst_gl_context_get_display(glcontext));
          gst_element_set_context(GST_ELEMENT(msg->src), gstcontext);
          gst_context_unref(gstcontext);
          gst_object_unref(glcontext);
        }
        gst_object_unref(vosink);
      }
    }
  }
  return GST_BUS_PASS;
}

GstFlowReturn App::OnNewSample(GstElement *element) {
  GstSample *sample = NULL;
  g_signal_emit_by_name(element, "pull-sample", &sample);
  CHECK_NOTNULL(sample);

  mutex_.Lock();
  bool use_runner = use_runner_;
  mutex_.Unlock();

  if (use_runner) {
    // Note: don't ever memcpy the input buffer if avoidable!
    coral::PipelineTensor tensor;
    tensor.data.data = allocator_.wrap(sample);
    tensor.bytes = input_bytes_;
    tensor.type = kTfLiteUInt8;

    CHECK_EQ(runner_->Push({tensor}), true);
    mutex_.Lock();
    start_times_.push_front(std::chrono::steady_clock::now());
    mutex_.Unlock();
  } else {
    // TODO
    CHECK(false) << "Not implemented";
    gst_sample_unref(sample);
  }

  return GST_FLOW_OK;
}

void App::ConsumeRunner() {
  std::vector<coral::PipelineTensor> output_tensors;
  std::chrono::time_point<std::chrono::system_clock> start;
  start = std::chrono::system_clock::now();
  while (runner_->Pop(&output_tensors)) {
    CHECK_EQ(output_tensors.size(), static_cast<size_t>(1));

    mutex_.Lock();
    CHECK(!start_times_.empty());
    auto start_time = start_times_.back();
    start_times_.pop_back();
    mutex_.Unlock();

    std::chrono::duration<double, std::milli> duration_ms =
        std::chrono::steady_clock::now() - start_time;
    const uint8_t *tensor_data =
        static_cast<uint8_t *>(output_tensors[0].data.data);
    HandleOutput(tensor_data, duration_ms.count());

    coral::FreeTensors(output_tensors, runner_->GetOutputTensorAllocator());
    output_tensors.clear();
  }
}

void App::HandleOutput(const uint8_t *data, double duration_ms) {
  std::vector<float> results(output_bytes_);
  for (size_t i = 0; i < results.size(); i++) {
    results[i] = (data[i] - output_zero_point_) * output_scale_;
  }
  const auto max_element = std::max_element(results.begin(), results.end());
  const size_t max_index =
      static_cast<size_t>(std::distance(results.begin(), max_element));
  const float max_score = results[max_index];

  CHECK(max_index >= 0 && max_index < labels_.size())
      << "Inference result label index out of bounds";
  std::string max_label = labels_.at(max_index);

  mutex_.Lock();
  size_t frame_number = ++output_frames_;
  mutex_.Unlock();

  if (glsink_) {
    if (!video_width_ || !video_height_) {
      // Get source video resolution.
      GstPad *pad = CHECK_NOTNULL(gst_element_get_static_pad(glsink_, "sink"));
      GstCaps *caps = CHECK_NOTNULL(gst_pad_get_current_caps(pad));
      GstStructure *s = gst_caps_get_structure(caps, 0);
      CHECK(gst_structure_get_int(s, "width", &video_width_));
      CHECK(gst_structure_get_int(s, "height", &video_height_));
      gst_caps_unref(caps);
      gst_object_unref(pad);
      std::cout << absl::StrFormat("Source video resolution: %dx%d",
                                   video_width_, video_height_)
                << std::endl;
    }
    const int font_size = kFontSizePercent * video_height_ + 1;
    int row_height = font_size + 1;
    std::string header =
        absl::Substitute(kSvgHeader, video_width_, video_height_);
    std::string styles = absl::Substitute(kSvgStyles, font_size);
    std::string mode = absl::Substitute(
        kSvgText, 1, 1 * row_height,
        absl::StrFormat("Mode: %s", use_runner_ ? "Pipelined" : "Single TPU"),
        "");
    std::string latency =
        absl::Substitute(kSvgText, 1, 2 * row_height,
                         absl::StrFormat("Latency: %.2f ms", duration_ms), "");
    std::string score =
        absl::Substitute(kSvgText, 1, video_height_ - 5 - row_height,
                         absl::StrFormat("Score: %f", max_score), "");
    std::string label =
        absl::Substitute(kSvgText, 1, video_height_ - 5,
                         absl::StrFormat("Label: %s", max_label), "");
    std::string counter = absl::Substitute(
        kSvgText, video_width_ - 1, row_height,
        absl::StrFormat("Frame: %zu", frame_number), " text_ra");
    std::string svg = absl::StrCat(header, styles, mode, label, score, latency,
                                   counter, kSvgFooter);
    g_object_set(G_OBJECT(glsink_), "svg", svg.c_str(), NULL);
  } else {
    // TODO print something useful in headless mode.
    CHECK(false) << "not implemented";
  }
}

void App::Run() {
  // TODO: Switch modes with keyboard
  mutex_.Lock();
  use_runner_ = absl::GetFlag(FLAGS_use_multiple_edgetpu);
  loop_ = absl::GetFlag(FLAGS_loop);
  mutex_.Unlock();

  InitializeInference();
  InitializeGstreamer();

  if (absl::GetFlag(FLAGS_visualize)) {
    InitializeGtkWindow();
  }

  g_unix_signal_add(SIGINT,
                    reinterpret_cast<GSourceFunc>(+[](App *self) -> guint {
                      self->Quit();
                      return G_SOURCE_CONTINUE;
                    }),
                    this);

  std::thread consumer(&App::ConsumeRunner, this);
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  mutex_.Lock();
  running_ = true;
  mutex_.Unlock();
  gtk_main();
  gst_element_set_state(pipeline_, GST_STATE_NULL);
  runner_->Push({});
  consumer.join();
  DestroyGtkWindow();
}

void App::Quit() {
  mutex_.Lock();
  bool running = running_;
  running_ = false;
  mutex_.Unlock();

  if (running) {
    gtk_main_quit();
  }
}

int main(int argc, char *argv[]) {
  absl::ParseCommandLine(argc, argv);
  google::InitGoogleLogging(argv[0]);

  App app;
  app.Run();

  return 0;
}
