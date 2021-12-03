// Copyright 2019 The MediaPipe Authors.
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
//
// A utility to extract iris depth from a single image of face using the graph
// mediapipe/graphs/iris_tracking/iris_depth_cpu.pbtxt.
#include <cstdlib>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
// #include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
// GPU headers
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/deps/status_macros.h"

constexpr char kInputStream[] = "input_image";
constexpr char kOutputImageStream[] = "output_image";
constexpr char kOutputFaceLandmarks[] = "multi_face_landmarks";
constexpr char kWindowName[] = "WinnerFaceMesh";
 constexpr char kCalculatorGraphConfigFile[] =
     "mediapipe/graphs/winner_face_mesh/wnr_face_mesh_image_gpu.pbtxt";
// constexpr char kCalculatorGraphConfigFile[] =
//     "mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt";
constexpr float kMicrosPerSecond = 1e6;

ABSL_FLAG(std::string, input_image_path, "",
          "Ruta completa de la imagen a cargar. "
          "Si no se provee falla la ejecución.");
ABSL_FLAG(std::string, output_image_path, "",
          "Ruta completa donde se guardará el archivo (.jpg). "
          "Si no se provee se mostrará en una ventana.");

namespace {

  absl::StatusOr<std::string> ReadFileToString(const std::string& file_path) {
    std::string contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(file_path, &contents));
    return contents;
  }

absl::Status RunMPPGraph() {
  // Donde se lee la configuracion del graph
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      kCalculatorGraphConfigFile, 
      &calculator_graph_config_contents)
  );
  LOG(INFO) << "Obteniendo el contenido del graph config: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Inicializando el calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Cargando la imagen.";
  ASSIGN_OR_RETURN(const std::string& raw_image,
                   ReadFileToString(absl::GetFlag(FLAGS_input_image_path)));

    LOG(INFO) << "Iniciando calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputImageStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller pollerLandMarks,
                   graph.AddOutputStreamPoller(kOutputFaceLandmarks));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  const std::vector<char> contents_vector(raw_image.begin(), raw_image.end());
  cv::Mat decoded_mat;
  decoded_mat = cv::imdecode(contents_vector, cv::IMREAD_UNCHANGED);
  
  mediapipe::ImageFormat::Format image_format = mediapipe::ImageFormat::UNKNOWN;
  cv::Mat output_mat;
  switch (decoded_mat.channels()) {
    case 1:
      image_format = mediapipe::ImageFormat::GRAY8;
      output_mat = decoded_mat;
      break;
    case 3:
      image_format = mediapipe::ImageFormat::SRGB;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGB);
      break;
    case 4:
      image_format = mediapipe::ImageFormat::SRGBA;
      cv::cvtColor(decoded_mat, output_mat, cv::COLOR_BGR2RGBA);
      break;
    default:
      return mediapipe::FailedPreconditionErrorBuilder(MEDIAPIPE_LOC)
             << "Unsupported number of channels: " << decoded_mat.channels();
  }
  std::unique_ptr<mediapipe::ImageFrame> input_frame = absl::make_unique<mediapipe::ImageFrame>(
      image_format, decoded_mat.size().width, decoded_mat.size().height,
      mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
  output_mat.copyTo(mediapipe::formats::MatView(input_frame.get()));

  // Se envía el packet dentro del graph.
  const size_t frame_timestamp_us = (double)cv::getTickCount() /
                                   (double)cv::getTickFrequency() *
                                   kMicrosPerSecond;
  MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return absl::OkStatus();
        }));

  // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    LOG(INFO) << "poller.QueueSize()" << poller.QueueSize();
    LOG(INFO) << "pollerLandMarkspoller.QueueSize()" << pollerLandMarks.QueueSize();
    if (!poller.Next(&packet)) {
      return absl::UnknownError(
          "Falló el obtener el paquete desde el output stream.");
    }


    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> absl::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
              gpu_frame.format(), 0, gpu_helper.GetGlVersion());
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return absl::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    // cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    // if (output_frame_mat.channels() == 4)
    //   cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
    // else
    //   cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    // auto& output_frame = output_image_packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    const bool save_image = !absl::GetFlag(FLAGS_output_image_path).empty();
    if (save_image) {
      LOG(INFO) << "Guardando imagen al archivo ...";
      cv::imwrite(absl::GetFlag(FLAGS_output_image_path), output_frame_mat);
    } else {
      cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      cv::waitKey(0);
    }

  }

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Falló la ejecución del graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "¡¡¡Éxito!!!";
  }
  return EXIT_SUCCESS;
}
