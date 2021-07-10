
#include "mediapipe/framework/calculator_framework.h"
#include <iostream>

namespace mediapipe {

static constexpr char kImageSizeTag[] = "SIZE";

  class LogInputCalculator: public CalculatorBase {

    public:

      LogInputCalculator(){};
      ~LogInputCalculator(){};

      static mediapipe::Status GetContract(CalculatorContract* cc){
        cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
        return mediapipe::OkStatus();
      }

      mediapipe::Status Open(CalculatorContext* cc){

          return mediapipe::OkStatus();
      }

      mediapipe::Status Process(CalculatorContext* cc){

        if (cc->Inputs().Tag(kImageSizeTag).IsEmpty()) {
          return absl::OkStatus();
        }
        
        const std::pair<int, int> image_size =
          cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

        std::cout << "first " << image_size.first << std::endl;// >> image_size.first;
        std::cout << "second " << image_size.second << std::endl;// >> image_size.second;

        return mediapipe::OkStatus();
      }

      mediapipe::Status Close(CalculatorContext* cc){

          return mediapipe::OkStatus();
      }
  };

  REGISTER_CALCULATOR(LogInputCalculator);
}
