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

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "wnr_face_mesh_image_gpu",
    srcs = ["wnr_face_mesh_image_gpu.cc"],
    deps = [
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",


        "//mediapipe/framework:calculator_framework",
        # "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        # "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",

        "//mediapipe/graphs/winner_face_mesh:wnr_face_mesh_image",

        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",

        # TEST ONLY :D :(
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
    ],
)

cc_library(
    name = "wnr_daemon",
    srcs = ["wnr_daemon.cc"],
    hdrs = ["wnr_daemon.h"],
    deps = [
        "//mediapipe/framework/port:logging",
    ]
)

cc_library(
    name = "wnr_msg_exchange",
    hdrs = ["wnr_msg_exchange.h"]
)

cc_binary(
    name = "winner_face_mesh_image",
    srcs = ["wnr_face_mesh_image.cc"],
    deps = [
        ":wnr_daemon",
        ":wnr_msg_exchange",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/graphs/winner_face_mesh:wnr_face_mesh_image",
        # "//mediapipe/graphs/winner_face_mesh:wnr_face_mesh_image_v2",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "winner_face_mesh_video",
    deps = [
        "//mediapipe/examples/desktop/winner_face_mesh:simple_run_graph_main",
        "//mediapipe/graphs/winner_face_mesh:desktop_calculators",
    ],
)

cc_library(
    name = "simple_run_graph_main",
    srcs = ["simple_run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
)