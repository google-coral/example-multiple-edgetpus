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
#include <gst/allocators/gstdmabuf.h>
#include <gst/gl/gl.h>
#include <gst/gst.h>
#include <gtk/gtk.h>
#include <sys/mman.h>
#include <termios.h>

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
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/pipeline/utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

ABSL_FLAG(std::string, model_path, "./test_data/inception_v3_299_quant",
          "Path to the tflite model base.");

ABSL_FLAG(std::string, video_path, "", "Path to the video file.");

ABSL_FLAG(std::string, labels_path, "./test_data/imagenet_labels.txt",
          "Path to labels file.");

ABSL_FLAG(bool, use_multiple_edgetpu, true,
          "Using PipelinedModelRunner or tflite::Interpreter.");

ABSL_FLAG(int, num_segments, 2, "Number of segments (Edge TPUs).");

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

// Pipelining has no concept of rate limiting (yet). For the Dev Board there is
// already an implicit rate limit due to the decode time of the VPU. As such we
// allow a large number of frames to be queued (the VPU will actually do the
// rate limiting). For others (for example a fast k8 machine), the decode could
// potentially be much faster than the inference - we use a mutex to block so
// that only one frame is sent at a time.
const int kDevBoardMaxFrames = 100;
const int kOthersMaxFrames = 1;

static bool IsDevBoard() {
  std::ifstream file("/sys/firmware/devicetree/base/model");
  std::string line;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      if (line.find("MX8MQ") != std::string::npos) {
        return true;
      }
    }
  }

  return false;
}

class AverageCounter {
 public:
  AverageCounter(size_t size) : size_(size) {}
  double PushValue(double value) {
    data_.push_front(value);
    while (data_.size() > size_) {
      data_.pop_back();
    }
    value = 0;
    for (double v : data_) {
      value += v;
    }
    return value / data_.size();
  }

 private:
  size_t size_;
  std::deque<double> data_;
};

class DmaBuffer : public coral::Buffer {
 public:
  DmaBuffer(GstSample *sample, size_t requested_bytes)
      : sample_(CHECK_NOTNULL(sample)), requested_bytes_(requested_bytes) {}

  // For cases where we can't use DMABuf (like most x64 systems), return the
  // sample. This allows the pipeline runner to see there is no file descriptor
  // and instead rely on inefficient CPU mapping. Ideally this can optimized in
  // the future.
  void *ptr() override {
    GstMapInfo info;
    GstBuffer *buffer = CHECK_NOTNULL(gst_sample_get_buffer(sample_));
    CHECK(gst_buffer_map(buffer, &info, GST_MAP_READ));
    data_ = reinterpret_cast<void *>(info.data);
    return data_;
  }

  void *MapToHost() override {
    if (!handle_) {
      handle_ = mmap(nullptr, requested_bytes_, PROT_READ, MAP_PRIVATE, fd(),
                     /*offset=*/0);
      if (handle_ == MAP_FAILED) {
        handle_ = nullptr;
      }
    }
    return handle_;
  }

  bool UnmapFromHost() override {
    if (munmap(handle_, requested_bytes_) != 0) {
      return false;
    }
    return true;
  }

  int fd() {
    if (fd_ == -1) {
      GstBuffer *buf = CHECK_NOTNULL(gst_sample_get_buffer(sample_));
      GstMemory *mem = gst_buffer_peek_memory(buf, 0);
      if (gst_is_dmabuf_memory(mem)) {
        fd_ = gst_dmabuf_memory_get_fd(mem);
      }
    }
    return fd_;
  }

 private:
  friend class DmaAllocator;

  GstSample *sample_ = nullptr;
  size_t requested_bytes_ = 0;

  // DMA Buffer variables
  int fd_ = -1;
  void *handle_ = nullptr;

  // Legacy CPU variables
  void *data_ = nullptr;
};

class DmaAllocator : public coral::Allocator {
 public:
  DmaAllocator() = default;
  DmaAllocator(GstElement *sink) : sink_(CHECK_NOTNULL(sink)) {}

  coral::Buffer *Alloc(size_t size_bytes) override {
    GstSample *sample;
    g_signal_emit_by_name(sink_, "pull-sample", &sample);
    return new DmaBuffer(sample, size_bytes);
  }

  void Free(coral::Buffer *buffer) override {
    auto *sample = static_cast<DmaBuffer *>(buffer)->sample_;
    if (sample) {
      gst_sample_unref(sample);
    }

    delete buffer;
  }

  void UpdateAppSink(GstElement *sink) {
    CHECK_NOTNULL(sink);
    sink_ = sink;
  }

 private:
  GstElement *sink_ = nullptr;
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
  void OnStdinReady(GIOChannel *ioc);
  void ConsumeRunner();
  void FeedInterpreter();
  void HandleOutput(const uint8_t *data, double inference_ms, double intra_ms);
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
  GstSample *sample_ = nullptr;
  DmaAllocator allocator_;
  AverageCounter inference_times_;
  AverageCounter intra_times_;
  std::chrono::time_point<std::chrono::steady_clock> last_inference_;
  int input_width_;
  int input_height_;
  size_t input_bytes_;
  float output_scale_;
  int32_t output_zero_point_;
  size_t output_bytes_;
  size_t output_frames_ = 0;
  size_t line_length_ = 0;
  int video_width_ = 0;
  int video_height_ = 0;
  bool use_runner_;
  bool running_;
  bool loop_;
  absl::Mutex mutex_;
  absl::CondVar cond_;
  bool is_dev_board_;
  int max_queue_size_;
  int frames_in_tpu_queue_ = 0;
};

App::App()
    : inference_times_(AverageCounter(100)), intra_times_(AverageCounter(100)) {
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
  size_t pci_index = num_segments - 1;
  for (size_t i = 0; i < num_segments; ++i) {
    edgetpu_contexts_[i] =
        CHECK_NOTNULL(edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            all_tpus[i].type, all_tpus[i].path));
    std::cout << "Device " << all_tpus[i].path << " opened" << std::endl;
    if (all_tpus[i].type == edgetpu::DeviceType::kApexPci) {
      pci_index = i;
    }
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
                                       edgetpu_contexts_[pci_index].get());
  std::cout << "Full model " << full_model_path << " loaded for "
            << all_tpus[pci_index].path << std::endl;

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
  std::string overlay_src;
  if (is_dev_board_) {
    overlay_src =
        "filesrc location=$0 ! decodebin ! glupload ! tee name=t\n"
        " t. ! queue ! glsvgoverlaysink name=glsink\n"
        " t. ! queue ! "
        "glfilterbin filter=glbox ! "
        "video/x-raw,width=$1,height=$2,format=RGB\n"
        "    ! queue max-size-buffers=1 leaky=downstream"
        " ! appsink name=appsink emit-signals=true";
  } else {
    overlay_src =
        "filesrc location=$0 ! decodebin ! tee name=t\n"
        " t. ! queue ! videoconvert ! autovideosink\n"
        "    ! queue max-size-buffers=1 leaky=no"
        " t. ! queue ! videoconvert ! videoscale ! "
        "video/x-raw,width=$1,height=$2,format=RGB\n"
        "    ! queue max-size-buffers=1 leaky=no"
        " ! appsink name=appsink emit-signals=true";
  }
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
  if (is_dev_board_) {
    glsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "glsink");
  }

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
  allocator_.UpdateAppSink(appsink);
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

void App::OnStdinReady(GIOChannel *ioc) {
  gchar c;
  GIOStatus ret;
  gsize read;
  GError *error = NULL;

  do {
    switch (ret = g_io_channel_read_chars(ioc, &c, 1, &read, &error)) {
      case G_IO_STATUS_NORMAL:
        switch (c) {
          case ' ':
            mutex_.Lock();
            use_runner_ = !use_runner_;
            mutex_.Unlock();
            break;
          case 'q':
          case 'Q':
            Quit();
            break;
        }
        break;
      case G_IO_STATUS_ERROR:
        std::cout << absl::StrFormat("IO error: %s", error->message)
                  << std::endl;
        g_error_free(error);
        Quit();
        break;
      case G_IO_STATUS_EOF:
      case G_IO_STATUS_AGAIN:
        break;
    }
  } while (read);
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
  mutex_.Lock();
  bool use_runner = use_runner_;
  mutex_.Unlock();

  // Note: don't ever memcpy the input buffer if avoidable. In both paths below
  // care is taken to avoid this performance bottleneck.
  if (use_runner) {
    coral::PipelineTensor input_buffer;
    const TfLiteTensor *input_tensor = interpreter_->input_tensor(0);
    input_buffer.buffer =
        runner_->GetInputTensorAllocator()->Alloc(input_tensor->bytes);
    input_buffer.type = input_tensor->type;
    input_buffer.bytes = input_tensor->bytes;

    CHECK_EQ(runner_->Push({input_buffer}), true);
    mutex_.Lock();
    frames_in_tpu_queue_++;
    start_times_.push_front(std::chrono::steady_clock::now());
    while (frames_in_tpu_queue_ >= max_queue_size_) {
      cond_.Wait(&mutex_);
    }
    mutex_.Unlock();
  } else {
    mutex_.Lock();
    GstSample *sample = NULL;
    g_signal_emit_by_name(element, "pull-sample", &sample);
    CHECK_NOTNULL(sample);
    if (sample_) {
      gst_sample_unref(sample_);
    }
    sample_ = sample;
    cond_.SignalAll();
    mutex_.Unlock();
  }

  return GST_FLOW_OK;
}

void App::ConsumeRunner() {
  std::vector<coral::PipelineTensor> output_tensors;
  while (runner_->Pop(&output_tensors)) {
    CHECK_EQ(output_tensors.size(), static_cast<size_t>(1));

    mutex_.Lock();
    frames_in_tpu_queue_--;
    cond_.SignalAll();
    CHECK(!start_times_.empty());
    auto start_time = start_times_.back();
    start_times_.pop_back();
    std::chrono::duration<double, std::milli> inference_ms =
        std::chrono::steady_clock::now() - start_time;
    std::chrono::duration<double, std::milli> intra_ms =
        std::chrono::steady_clock::now() - last_inference_;
    last_inference_ = std::chrono::steady_clock::now();
    mutex_.Unlock();

    const uint8_t *tensor_data =
        static_cast<uint8_t *>(output_tensors[0].buffer->ptr());
    HandleOutput(tensor_data, inference_ms.count(), intra_ms.count());

    coral::FreeTensors(output_tensors, runner_->GetOutputTensorAllocator());
    output_tensors.clear();
  }
}

void App::FeedInterpreter() {
  bool running;
  do {
    GstSample *sample;
    mutex_.Lock();
    while (!sample_ && running_) {
      cond_.Wait(&mutex_);
    }
    sample = sample_;
    sample_ = nullptr;
    running = running_;
    mutex_.Unlock();

    if (running && sample) {
      // TODO: Support passing dma-buf handles without mapping.
      const auto start_time = std::chrono::steady_clock::now();
      GstMapInfo info;
      GstBuffer *buffer = CHECK_NOTNULL(gst_sample_get_buffer(sample));
      CHECK(gst_buffer_map(buffer, &info, GST_MAP_READ));

      const auto *tensor = CHECK_NOTNULL(interpreter_->input_tensor(0));
      auto quantization = tensor->quantization;
      if (quantization.type != kTfLiteNoQuantization) {
        CHECK_EQ(quantization.type, kTfLiteAffineQuantization);
        auto *src =
            reinterpret_cast<TfLiteAffineQuantization *>(quantization.params);
        auto *copy = CHECK_NOTNULL(static_cast<TfLiteAffineQuantization *>(
            malloc(sizeof(TfLiteAffineQuantization))));
        copy->scale = CHECK_NOTNULL(static_cast<TfLiteFloatArray *>(
            malloc(TfLiteFloatArrayGetSizeInBytes(src->scale->size))));
        copy->scale->size = src->scale->size;
        std::memcpy(copy->scale->data, src->scale->data,
                    TfLiteFloatArrayGetSizeInBytes(src->scale->size));
        copy->zero_point = TfLiteIntArrayCopy(src->zero_point);
        copy->quantized_dimension = src->quantized_dimension;
        quantization.params = copy;
      }

      const auto shape =
          absl::Span<const int>(tensor->dims->data, tensor->dims->size);
      CHECK_EQ(interpreter_->SetTensorParametersReadOnly(
                   interpreter_->inputs()[0], tensor->type, tensor->name,
                   std::vector<int>(shape.begin(), shape.end()), quantization,
                   reinterpret_cast<const char *>(info.data), input_bytes_),
               kTfLiteOk);
      CHECK_EQ(tensor->data.raw, reinterpret_cast<char *>(info.data));
      gst_buffer_unmap(buffer, &info);

      CHECK_EQ(interpreter_->Invoke(), kTfLiteOk);
      std::chrono::duration<double, std::milli> inference_ms =
          std::chrono::steady_clock::now() - start_time;
      mutex_.Lock();
      std::chrono::duration<double, std::milli> intra_ms =
          std::chrono::steady_clock::now() - last_inference_;
      last_inference_ = std::chrono::steady_clock::now();
      mutex_.Unlock();

      uint8_t *output_data =
          CHECK_NOTNULL(interpreter_->typed_output_tensor<uint8_t>(0));
      HandleOutput(output_data, inference_ms.count(), intra_ms.count());
    }
    if (sample) {
      gst_sample_unref(sample);
    }
  } while (running);

  mutex_.Lock();
  if (sample_) {
    gst_sample_unref(sample_);
    sample_ = nullptr;
  }
  mutex_.Unlock();
}

void App::HandleOutput(const uint8_t *data, double inference_ms,
                       double intra_ms) {
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
  double avg_inference_ms = inference_times_.PushValue(inference_ms);
  double avg_intra_ms = intra_times_.PushValue(intra_ms);
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
        absl::StrFormat("Mode: %s", use_runner_ ? "Pipelined" : "SingleTPU"),
        "");
    std::string fps =
        absl::Substitute(kSvgText, 1, 2 * row_height,
                         absl::StrFormat("Processing FPS: %.1f (%.1f)",
                                         1000 / intra_ms, 1000 / avg_intra_ms),
                         "");
    std::string latency =
        absl::Substitute(kSvgText, 1, 3 * row_height,
                         absl::StrFormat("Inference: %.1f (%.1f) ms",
                                         inference_ms, avg_inference_ms),
                         "");
    std::string score =
        absl::Substitute(kSvgText, 1, video_height_ - 5 - row_height,
                         absl::StrFormat("Score: %f", max_score), "");
    std::string label =
        absl::Substitute(kSvgText, 1, video_height_ - 5,
                         absl::StrFormat("Label: %s", max_label), "");
    std::string counter = absl::Substitute(
        kSvgText, video_width_ - 1, row_height,
        absl::StrFormat("Frame: %zu", frame_number), " text_ra");
    std::string svg = absl::StrCat(header, styles, mode, label, score, fps,
                                   latency, counter, kSvgFooter);
    g_object_set(G_OBJECT(glsink_), "svg", svg.c_str(), NULL);
  } else {
    std::string line = absl::StrFormat(
        "\r%s: frame %zu, processing fps %.1f/%.1f, latency (ms) %.1f/%.1f, "
        "score %.2f label '%s'",
        use_runner_ ? "Pipelined" : "SingleTPU", frame_number, 1000 / intra_ms,
        1000 / avg_intra_ms, inference_ms, avg_inference_ms, max_score,
        max_label);
    absl::MutexLock lock(&mutex_);
    line_length_ = std::max(line_length_, line.size());
    std::cout << std::setw(line_length_) << line << std::flush;
  }
}

void App::Run() {
  mutex_.Lock();
  use_runner_ = absl::GetFlag(FLAGS_use_multiple_edgetpu);
  loop_ = absl::GetFlag(FLAGS_loop);
  mutex_.Unlock();

  is_dev_board_ = IsDevBoard();
  max_queue_size_ = is_dev_board_ ? kDevBoardMaxFrames : kOthersMaxFrames;

  InitializeInference();
  InitializeGstreamer();

  // Only visualize on the Dev Board for now.
  if (absl::GetFlag(FLAGS_visualize) && is_dev_board_) {
    InitializeGtkWindow();
  }

  g_unix_signal_add(SIGINT,
                    reinterpret_cast<GSourceFunc>(+[](App *self) -> guint {
                      self->Quit();
                      return G_SOURCE_CONTINUE;
                    }),
                    this);

  GIOChannel *ioc = nullptr;
  guint iow = 0;
  struct termios orig_attrs;
  if (isatty(STDIN_FILENO)) {
    struct termios attrs;
    tcgetattr(STDIN_FILENO, &orig_attrs);
    tcgetattr(STDIN_FILENO, &attrs);
    attrs.c_lflag &= ~(ICANON | ECHO);
    attrs.c_cc[VMIN] = 1;
    attrs.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &attrs);
    ioc = g_io_channel_unix_new(STDIN_FILENO);
    g_io_channel_set_flags(
        ioc,
        static_cast<GIOFlags>(g_io_channel_get_flags(ioc) | G_IO_FLAG_NONBLOCK),
        NULL);
    iow = g_io_add_watch(
        ioc, G_IO_IN,
        reinterpret_cast<GIOFunc>(
            +[](GIOChannel *ch, GIOCondition cond, App *self) -> gboolean {
              self->OnStdinReady(ch);
              return TRUE;
            }),
        this);
    std::cout << std::endl << "Press spacebar to switch modes" << std::endl;
    std::cout << "Press q to quit" << std::endl;
  }

  mutex_.Lock();
  running_ = true;
  last_inference_ = std::chrono::steady_clock::now();
  mutex_.Unlock();
  std::thread consumer(&App::ConsumeRunner, this);
  std::thread feeder(&App::FeedInterpreter, this);
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  gtk_main();

  if (ioc) {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_attrs);
    g_source_remove(iow);
    g_io_channel_unref(ioc);
  }
  gst_element_set_state(pipeline_, GST_STATE_NULL);
  mutex_.Lock();
  cond_.SignalAll();
  mutex_.Unlock();
  runner_->Push({});
  consumer.join();
  feeder.join();
  CHECK(!sample_);
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
