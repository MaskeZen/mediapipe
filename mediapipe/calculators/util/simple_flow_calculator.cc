// This implements a simple calculator which takes a video stream as input and 
// output it without applyiing any operations on its frames.

#include <vector>
#include <iostream>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class SimpleFlowCalculator : public CalculatorBase {
 public:



  static ::mediapipe::Status GetContract(CalculatorContract* cc) {

// Specify the expected types of inputs and outputs of a calculator in GetContract()
// When a graph is initialized, the framework calls a static method to verify if the 
// packet types of the connected input and outputs match the information in this specification

  RET_CHECK(cc->Inputs().HasTag("IMAGE") ^ cc->Inputs().HasTag("IMAGE_GPU"));
  RET_CHECK(cc->Outputs().HasTag("IMAGE") ^ cc->Outputs().HasTag("IMAGE_GPU"));
  std::cout << " GetContract() " << std::endl;

  if (cc->Inputs().HasTag("IMAGE")) {
    RET_CHECK(cc->Outputs().HasTag("IMAGE"));
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
  }


    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
   
      // After a graph starts, the framework calls Open(). The input side packets are available to 
      // the calculator at the point. Open() interprets the node configuration operations and prepares
      // the calculator's per-graph-run state. This function may also write packets to calculator outputs.
      // An error during Open() can terminate the graph run.
    
   std::cout << " Open() " << std::endl;
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    
    
    // For a calculator with inputs, the framework calls Process( ) repeatedly whenever at least one input
    // stream has a packet available. The framework by default guarntees that all inputs have the same times-
    // stamp. Multiple Process)_ calls can be invoked simultaneously when parallel execution is enabled. If an 
    // error occurs during Process(), the framework calls Close() and the graph run terminates.
    
    std::cout << " Process() " << std::endl;
    const auto& input_image = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();

    // converts mediapipe_imageFrame to OpenCV Mat
    cv::Mat input_mat = formats::MatView(&input_image);
    
    std::unique_ptr<ImageFrame> output_frame(
           new ImageFrame(input_image.Format(),input_image.Width(),input_image.Height())
    );
    
    cv::Mat output_mat = formats::MatView(output_frame.get());
    input_mat.copyTo(output_mat);
    
    cc->Outputs().Tag("IMAGE").Add(output_frame.release(),cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  }

};

REGISTER_CALCULATOR(SimpleFlowCalculator);
}  // namespace mediapipe