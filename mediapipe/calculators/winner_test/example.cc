
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

    class ExampleCalculator: public CalculatorBase {

        public:

        ExampleCalculator(){};
        ~ExampleCalculator(){};

        static ::mediapipe::Status GetContract(CalculatorContract* cc){

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Open(CalculatorContext* cc){

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Process(CalculatorContext* cc){

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Close(CalculatorContext* cc){

            return ::mediapipe::OkStatus();
        }
    };

    REGISTER_CALCULATOR(ExampleCalculator);

}
