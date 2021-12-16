#!/bin/bash

inputPath=$1

cd ~/repo/mediapipe

procesarArchivo() {
    input_image_path=$1
    output_image_path="./img/output_image_$(date +%Y%m%d_%H%M%S).jpg"

    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/winner_face_mesh/winner_face_mesh_image \
        --input_image_path=$input_image_path --output_image_path=$output_image_path
}

if [[ -d $inputPath ]]; then
    echo "Se procesará el directorio $inputPath"
    for file in $inputPath/*.jpg; do
        echo "Procesando archivo $file ------------------------------------------------"
        procesarArchivo $file
    done
else
    echo "Input path is a file"
    procesarArchivo $inputPath
fi

cd -