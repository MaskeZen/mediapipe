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
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
// #include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "wnr_daemon.h"
#include <thread>

// IPC -------------
#include <sys/ipc.h>
// shared memory
#include <sys/shm.h>
#include "wnr_msg_exchange.h"
#include <stdio.h>
#include <string>
#include <cstring>
// -----------------

#include <Eigen/Geometry>

// bool closeEnough(const float& a, const float& b, const float& epsilon = std::numeric_limits<float>::epsilon()) {
//     return (epsilon > std::abs(a - b));
// }

// cv::Vec3f getEulerAngles(cv::Mat &R) {

//     //check for gimbal lock
//     if (closeEnough(R.at<double>(0, 2), -1.0f)) {
//         float x = 0; //gimbal lock, value of x doesn't matter
//         float y = M_PI / 2;
//         float z = x + atan2(R.at<double>(1, 0), R.at<double>(2, 0));
//         return { x, y, z };
//     } else if (closeEnough(R.at<double>(0, 2), 1.0f)) {
//         float x = 0;
//         float y = -M_PI / 2;
//         float z = -x + atan2(-R.at<double>(1, 0), -R.at<double>(2, 0));
//         return { x, y, z };
//     } else { //two solutions exist
//         float x1 = -asin(R.at<double>(0, 2));
//         float x2 = M_PI - x1;

//         float y1 = atan2(R.at<double>(1, 2) / cos(x1), R.at<double>(2, 2) / cos(x1));
//         float y2 = atan2(R.at<double>(1, 2) / cos(x2), R.at<double>(2, 2) / cos(x2));

//         float z1 = atan2(R.at<double>(0, 1) / cos(x1), R.at<double>(0, 0) / cos(x1));
//         float z2 = atan2(R.at<double>(0, 1) / cos(x2), R.at<double>(0, 0) / cos(x2));

//         //choose one solution to return
//         //for example the "shortest" rotation
//         if ((std::abs(x1) + std::abs(y1) + std::abs(z1)) <= (std::abs(x2) + std::abs(y2) + std::abs(z2))) {
//             return { x1, y1, z1 };
//         } else {
//             return { x2, y2, z2 };
//         }
//     }
// }

constexpr char kInputStream[] = "input_image_bytes";
constexpr char kOutputImageStream[] = "output_image";
constexpr char kOutputFaceGeometry[] = "multi_face_geometry";

constexpr char kWindowName[] = "WinnerFaceMesh";
constexpr char kCalculatorGraphConfigFile[] =
    "mediapipe/graphs/winner_face_mesh/wnr_face_mesh_image_v2.pbtxt";
constexpr float kMicrosPerSecond = 1e6;

ABSL_FLAG(std::string, input_image_path, "",
          "Ruta completa de la imagen a cargar. "
          "Si no se provee falla la ejecución.");
ABSL_FLAG(std::string, output_image_path, "",
          "Ruta completa donde se guardará el archivo (.jpg). "
          "Si no se provee se mostrará en una ventana.");

namespace
{

  void printEugerAnglesResult(cv::Vec3f eulerAngles, bool convertToDegrees = true)
  {
    float pitch = eulerAngles[0];
    float yaw = eulerAngles[1];
    float roll = eulerAngles[2];
    LOG(INFO) << "getEulerAngles"
              << "x: " << pitch << " y: " << yaw << " z: " << roll;

    if (convertToDegrees)
    {
      pitch = pitch * 180 / M_PI;
      yaw = yaw * 180 / M_PI;
      roll = roll * 180 / M_PI;
    }

    LOG(INFO) << " Pitch: " << pitch << " Yaw: " << yaw << " Roll: " << roll;
  }

  absl::StatusOr<std::string> ReadFileToString(const std::string &file_path)
  {
    std::string contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(file_path, &contents));
    return contents;
  }
  
  absl::StatusOr<std::string> ReturnString(std::string packet_image)
  {
    std::string contents = packet_image;
    return contents;
  }

  // Chequea si es una rotationmatrix válida.
  bool isRotationMatrix(cv::Mat &R)
  {
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    bool esRM = norm(I, shouldBeIdentity) < 1e-6;
    if (esRM == false)
    {
      LOG(INFO) << "xxxxxxxxxxxxx No es una rotation matrix.";
    }

    return esRM;
  }

  // Calcula rotation matrix a euler angles
  // http://nghiaho.com/?page_id=846
  // https://learnopencv.com/rotation-matrix-to-euler-angles/
  cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
  {
    assert(isRotationMatrix(R));
    // isRotationMatrix(R);

    float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));
    // float sy = sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) + R.at<double>(2, 2) * R.at<double>(2, 2));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
      // LOG(INFO) << "xxxxxxxxxxxxx No es singular.";
      x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
      y = atan2(-R.at<float>(2, 0), sy);
      z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    }
    else
    {
      // LOG(INFO) << "xxxxxxxxxxxxx Es singular.";
      x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
      y = atan2(-R.at<float>(2, 0), sy);
      z = 0;
    }

    return cv::Vec3f(x, y, z);
  }

  absl::Status ProcessImage(std::unique_ptr<mediapipe::CalculatorGraph> graph, std::string packet_image)
  {
    LOG(INFO) << "Se inicia el procesamiento de la imagen.";
    ASSIGN_OR_RETURN(const std::string raw_image,
                      ReturnString(packet_image));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller output_face_geometry_poller,
                     graph->AddOutputStreamPoller(kOutputFaceGeometry));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller output_image_poller,
                     graph->AddOutputStreamPoller(kOutputImageStream));
    MP_RETURN_IF_ERROR(graph->StartRun({}));

    // LOG(INFO) << "xxxxxxxxxxxxx raw_image:" << raw_image;

    // Se envía el packet dentro del graph.
    const size_t fake_timestamp_us = (double)cv::getTickCount() /
                                     (double)cv::getTickFrequency() *
                                     kMicrosPerSecond;
    MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
        kInputStream, mediapipe::MakePacket<std::string>(raw_image).At(
                          mediapipe::Timestamp(fake_timestamp_us))));

    // Get the graph result packets, or stop if that fails.
    mediapipe::Packet output_face_geometry_packet;
    if (!output_face_geometry_poller.Next(&output_face_geometry_packet))
    {
      return absl::UnknownError(
          "Falló el obtener el paquete desde el output stream 'multi_face_geometry'.");
    }
    // LOG(INFO) << "Se intentará obtener el FaceGeometry";
    auto &face_geometry_vector = output_face_geometry_packet.Get<std::vector<mediapipe::face_geometry::FaceGeometry>>();

    auto &face_geometry = face_geometry_vector[0];
    // LOG(INFO) << "FaceGeometry OK!";
    // LOG(INFO) << "Se intentará obtener el MatrixData";
    const mediapipe::MatrixData &pose_transform_matrix = face_geometry.pose_transform_matrix();
    // LOG(INFO) << "MatrixData OK!";
    // LOG(INFO) << "MatrixData rows(): " << pose_transform_matrix.rows();
    // LOG(INFO) << "MatrixData cols(): " << pose_transform_matrix.cols();
    //  ==----------------------------------------------------------------------------------==
    // LOG(INFO) << "MatrixData data(0): " << pose_transform_matrix.packed_data(0);
    // LOG(INFO) << "MatrixData data(1): " << pose_transform_matrix.packed_data(1);
    // LOG(INFO) << "MatrixData data(2): " << pose_transform_matrix.packed_data(2);
    // LOG(INFO) << "MatrixData data(3): " << pose_transform_matrix.packed_data(3);
    // LOG(INFO) << "MatrixData data(4): " << pose_transform_matrix.packed_data(4);
    // LOG(INFO) << "MatrixData data(5): " << pose_transform_matrix.packed_data(5);
    // LOG(INFO) << "MatrixData data(6): " << pose_transform_matrix.packed_data(6);
    // LOG(INFO) << "MatrixData data(7): " << pose_transform_matrix.packed_data(7);
    // LOG(INFO) << "MatrixData data(8): " << pose_transform_matrix.packed_data(8);
    // LOG(INFO) << "MatrixData data(9): " << pose_transform_matrix.packed_data(9);
    // LOG(INFO) << "MatrixData data(10): " << pose_transform_matrix.packed_data(10);
    // LOG(INFO) << "MatrixData data(11): " << pose_transform_matrix.packed_data(11);
    // LOG(INFO) << "MatrixData data(12): " << pose_transform_matrix.packed_data(12);
    // LOG(INFO) << "MatrixData data(13): " << pose_transform_matrix.packed_data(13);
    // LOG(INFO) << "MatrixData data(14): " << pose_transform_matrix.packed_data(14);
    // LOG(INFO) << "MatrixData data(15): " << pose_transform_matrix.packed_data(15);

    // pose_transform_matrix to cv::Mat
    // cv::Mat rotationMatrix(3, 3, 0);
    cv::Mat rotationMatrix = cv::Mat_<float>(3, 3);
    for (int column = 0; column < 4; column++)
    {
      if (column == 3)
      {
        break;
      }
      for (int row = 0; row < 4; row++)
      {
        if (row == 3)
        {
          break;
        }
        float value = pose_transform_matrix.packed_data(column * 4 + row);
        // LOG(INFO) << "Col: " << row << "Row: " << column << " valor: " << value;
        // Invierto column por row para que el valor de la matriz quede en la forma correcta.
        rotationMatrix.at<float>(column, row) = value;
        // rotationMatrix.column(column * 4) = value;
      }
    }

    // LOG(INFO) << "rotationMatrix: [" ;
    // LOG(INFO) << rotationMatrix.at<float>(0, 0) << ", " << rotationMatrix.at<float>(0, 1) << ", " << rotationMatrix.at<float>(0, 2);
    // LOG(INFO) << rotationMatrix.at<float>(1, 0) << ", " << rotationMatrix.at<float>(1, 1) << ", " << rotationMatrix.at<float>(1, 2);
    // LOG(INFO) << rotationMatrix.at<float>(2, 0) << ", " << rotationMatrix.at<float>(2, 1) << ", " << rotationMatrix.at<float>(2, 2);
    // LOG(INFO) << "]";

    // LOG(INFO) << "rotationMatrix.rows: " << rotationMatrix.rows;
    // LOG(INFO) << "rotationMatrix.cols: " << rotationMatrix.cols;

    // LOG(INFO) << "=============== rotationMatrixToEulerAngles ===============";
    printEugerAnglesResult(rotationMatrixToEulerAngles(rotationMatrix), true);
    // cv::Vec3f eulerAngles = rotationMatrixToEulerAngles(rotationMatrix);
    // LOG(INFO) << "=============== getEulerAngles ===============";
    // printEugerAnglesResult(getEulerAngles(rotationMatrix), true);
    // pt_matrix = [
    // 0.5567780137062073, 0.034023914486169815, 0.8299639821052551, 0,
    // -0.011918997392058372, 0.9993847608566284, -0.03297339752316475, 0,
    // -0.8305754661560059, 0.008466490544378757, 0.5568410754203796, 0,
    // -1.418548345565796, 6.790719509124756, -39.25355529785156, 1
    // ]

    mediapipe::Packet output_image_packet;
    if (!output_image_poller.Next(&output_image_packet))
    {
      return absl::UnknownError(
          "Falló el obtener el paquete desde el output stream 'output_image'.");
    }
    auto &output_frame = output_image_packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    const bool save_image = !absl::GetFlag(FLAGS_output_image_path).empty();
    if (save_image)
    {
      // LOG(INFO) << "Guardando imagen al archivo ...";
      cv::imwrite(absl::GetFlag(FLAGS_output_image_path), output_frame_mat);
    }
    else
    {
      cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      cv::waitKey(0);
    }

    // LOG(INFO) << "Finalizando...";
    MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
    return graph->WaitUntilDone();
  }

  absl::Status RunMPPGraph(std::string raw_image)
  {
    // Donde se lee la configuracion del graph
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        kCalculatorGraphConfigFile,
        &calculator_graph_config_contents));
    // LOG(INFO) << "Obteniendo el contenido del graph config: "
    //           << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    // LOG(INFO) << "Inicializando el calculator graph.";
    std::unique_ptr<mediapipe::CalculatorGraph> graph =
        absl::make_unique<mediapipe::CalculatorGraph>();
    MP_RETURN_IF_ERROR(graph->Initialize(config));

    return ProcessImage(std::move(graph), raw_image);
  }

  void reload()
  {
    LOG(INFO) << "Se recarga el demonio.";
  }

} // namespace

int main(int argc, char **argv)
{

  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  winnerPy::WnrDaemon &daemon = winnerPy::WnrDaemon::instance();
  daemon.setReloadFunction(reload);
  int last_msg_id = 0;
  LOG(INFO) << "Iniciando demonio.";
  int shmid = shmget(winnerPy::IMG_SHM_KEY, sizeof(winnerPy::datos_imagen), 0666|IPC_CREAT);
  LOG(INFO) << "Se comparte memoria de tamaño: " << sizeof(winnerPy::datos_imagen); 
  // int shmid = shmget(winnerPy::IMG_SHM_KEY, sizeof(winnerPy::datos_imagen),0666|IPC_CREAT);

  while (daemon.IsRunning())
  {
    // Se lee de la memoria compartida
		winnerPy::datos_imagen *datos = (winnerPy::datos_imagen*) shmat(shmid,(void*)0,0);
		if (datos == (void*)-1) {
			LOG(ERROR) << "No se pudo obtener la memoria compartida." << std::endl;
		 	//return 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(5000));
      continue;
		}

    LOG(INFO) << "Tamaño de la imagen: " << datos->img_size;
    int array_size = (datos->img_size * sizeof(char));
    LOG(INFO) << "Tamaño de la imagen array_size: " << array_size;

    // LOG(INFO) << "imagen: " << datos->imagen << std::endl;

    std::string raw_image = "";
    raw_image.resize(array_size);
    raw_image.clear();
    for (int i = 1; i <= array_size; i++)
    {
      raw_image += datos->imagen[i];
    }


    LOG(INFO) << "Tamaño de raw_image: " << raw_image.size();
    // raw_image.assign(datos->imagen_string.begin(), datos->imagen_string.size());
    // datos->imagen_string.copy(raw_image, datos->imagen_string.size(), 0);
    // const std::string raw_image(datos->imagen_string.begin(), datos->imagen_string.end());
    LOG(INFO) << "Se copia la imagen en la memoria compartida.";
        
    if (datos->msg_reply == 0 && datos->msg_id > last_msg_id) {
			last_msg_id = datos->msg_id;
			try {
        absl::Status run_status = RunMPPGraph(raw_image);
        if (!run_status.ok())
        {
          LOG(ERROR) << "Falló la ejecución del graph: " << run_status.message();
          // return EXIT_FAILURE;
        }
        else
        {
          LOG(INFO) << "¡¡¡Éxito!!!";
        }
				
			} catch  (const std::exception& e) {
				LOG(ERROR) << e.what();
			}
		} else {
      LOG(INFO) << "No hay nuevos mensajes.";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }

  return EXIT_SUCCESS;
}
