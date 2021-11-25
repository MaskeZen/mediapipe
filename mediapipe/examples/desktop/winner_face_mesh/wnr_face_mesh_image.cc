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
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_image_bytes";
constexpr char kOutputImageStream[] = "output_image";
constexpr char kWindowName[] = "WinnerFaceMesh";
constexpr char kCalculatorGraphConfigFile[] =
    "mediapipe/graphs/winner_face_mesh/wnr_face_mesh_image.pbtxt";
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

absl::Status ProcessImage(std::unique_ptr<mediapipe::CalculatorGraph> graph) {
  LOG(INFO) << "Cargando la imagen.";
  ASSIGN_OR_RETURN(const std::string raw_image,
                   ReadFileToString(absl::GetFlag(FLAGS_input_image_path)));

  LOG(INFO) << "Iniciando calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller output_image_poller,
                   graph->AddOutputStreamPoller(kOutputImageStream));
  MP_RETURN_IF_ERROR(graph->StartRun({}));

  // Se envía el packet dentro del graph.
  const size_t fake_timestamp_us = (double)cv::getTickCount() /
                                   (double)cv::getTickFrequency() *
                                   kMicrosPerSecond;
  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
      kInputStream, mediapipe::MakePacket<std::string>(raw_image).At(
                        mediapipe::Timestamp(fake_timestamp_us))));

  // Get the graph result packets, or stop if that fails.
  mediapipe::Packet output_image_packet;
  if (!output_image_poller.Next(&output_image_packet)) {
    return absl::UnknownError(
        "Falló el obtener el paquete desde el output stream 'output_image'.");
  }
  auto& output_frame = output_image_packet.Get<mediapipe::ImageFrame>();

  // Convert back to opencv for display or saving.
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
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

  LOG(INFO) << "Finalizando...";
  MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
  return graph->WaitUntilDone();
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
  std::unique_ptr<mediapipe::CalculatorGraph> graph =
      absl::make_unique<mediapipe::CalculatorGraph>();
  MP_RETURN_IF_ERROR(graph->Initialize(config));

  const bool load_image = !absl::GetFlag(FLAGS_input_image_path).empty();
  if (load_image) {
    return ProcessImage(std::move(graph));
  } else {
    return absl::InvalidArgumentError("No se encontró la imagen especificada.");
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
