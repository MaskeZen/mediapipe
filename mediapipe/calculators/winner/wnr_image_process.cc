
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"

#include <iostream>

namespace mediapipe {

  static constexpr char kImageSizeTag[] = "SIZE";
  static constexpr char kImageTag[] = "IMAGE";
  static constexpr char kImageOutTag[] = "IMAGE_OUT";

  class WnrImageProcess: public CalculatorBase {

    public:

      WnrImageProcess(){};
      ~WnrImageProcess(){};

      static mediapipe::Status GetContract(CalculatorContract* cc){
        cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
        cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
        cc->Outputs().Tag(kImageOutTag).Set<ImageFrame>();
        
        return mediapipe::OkStatus();
      }

      mediapipe::Status Open(CalculatorContext* cc){

          return mediapipe::OkStatus();
      }

      mediapipe::Status Process(CalculatorContext* cc){

        if (cc->Inputs().Tag(kImageSizeTag).IsEmpty()) {
          return absl::OkStatus();
        }
        if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
          return absl::OkStatus();
        }

        const std::pair<int, int> image_size =
          cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

        LOG(INFO) << "first " << image_size.first << std::endl;// >> image_size.first;
        LOG(INFO) << "second " << image_size.second << std::endl;// >> image_size.second;

        const auto& image =
          cc->Inputs().Tag(kImageTag).Get<ImageFrame>();

        // Obtiene unique_ptr de la ImageFrame
        // https://programmerclick.com/article/36771660317/
        // https://en.cppreference.com/w/cpp/memory/unique_ptr/unique_ptr
        auto image_out = absl::make_unique<ImageFrame>(image.Format(), image_size.first, image_size.second);

        // Se retorna la imagen procesada
        cc->Outputs().Tag(kImageOutTag).Add(image_out.release(), cc->InputTimestamp());

        return mediapipe::OkStatus();
      }

      mediapipe::Status Close(CalculatorContext* cc){

          return mediapipe::OkStatus();
      }
  };

  REGISTER_CALCULATOR(WnrImageProcess);
}
