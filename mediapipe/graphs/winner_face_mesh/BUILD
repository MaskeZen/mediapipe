# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "wnr_face_mesh_image",
    deps = [
        "//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/winner:wnr_image_process"
    ],
)

cc_library(
    name = "desktop_calculators",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/graphs/winner_face_mesh/subgraphs:face_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
    ],
)

cc_library(
    name = "process_image_calculators",
    deps = [
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/winner_test:log_input_calculator",
        "//mediapipe/modules/face_geometry:env_generator_calculator",
        "//mediapipe/graphs/winner_face_mesh/subgraphs:single_face_geometry_from_landmarks_gpu",
        # "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
    ]
)

cc_library(
    name = "log_image_calculators",
    deps = [
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/winner_test:log_input_calculator",
    ]
)
