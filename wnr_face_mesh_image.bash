#!/bin/bash

inputPath=$1

cd ~/repo/mediapipe

procesarArchivo() {
    input_image_path=$1
    output_image_path="./img/output_image_$(date +%Y%m%d_%H%M%S).jpg"

    echo "------------------------------------------------"
    echo "Procesando archivo $input_image_path"
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/winner_face_mesh/winner_face_mesh_image \
        --input_image_path=$input_image_path --output_image_path=$output_image_path
    echo "------------------------------------------------"
    echo "------------------------------------------------"
}

if [[ -d $inputPath ]]; then
    echo "Se procesará el directorio $inputPath:"
    echo "------------------------------------------------"
    for file in $inputPath/*; do
        procesarArchivo $file
    done
else
    echo "Se procesará el archivo $inputPath:"
    echo "------------------------------------------------"
    procesarArchivo $inputPath
fi

cd -