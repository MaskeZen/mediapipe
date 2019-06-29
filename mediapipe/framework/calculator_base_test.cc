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

#include "mediapipe/framework/calculator_base.h"

// TODO: Move protos in another CL after the C++ code migration.
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_context_manager.h"
#include "mediapipe/framework/calculator_registry.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/output_stream.h"
#include "mediapipe/framework/output_stream_manager.h"
#include "mediapipe/framework/output_stream_shard.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/tool/tag_map_helper.h"

namespace mediapipe {

namespace test_ns {

// A calculator which does nothing but accepts any number of input/output
// streams and input side packets.
class DeadEndCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
      cc->Inputs().Index(i).SetAny();
    }
    for (int i = 0; i < cc->Outputs().NumEntries(); ++i) {
      cc->Outputs().Index(i).SetAny();
    }
    for (int i = 0; i < cc->InputSidePackets().NumEntries(); ++i) {
      cc->InputSidePackets().Index(i).SetAny();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().NumEntries() > 0) {
      return ::mediapipe::OkStatus();
    } else {
      // This is a source calculator, but we don't produce any outputs.
      return tool::StatusStop();
    }
  }
};
REGISTER_CALCULATOR(::mediapipe::test_ns::DeadEndCalculator);

namespace whitelisted_ns {

class DeadCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};

}  // namespace whitelisted_ns
}  // namespace test_ns

class EndCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override {
    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(::mediapipe::EndCalculator);

namespace {

TEST(CalculatorTest, SourceProcessOrder) {
  internal::Collection<OutputStreamManager> output_stream_managers(
      tool::CreateTagMap(2).ValueOrDie());

  PacketType output0_type;
  PacketType output1_type;
  output0_type.SetAny();
  output1_type.SetAny();

  MEDIAPIPE_ASSERT_OK(
      output_stream_managers.Index(0).Initialize("output0", &output0_type));
  MEDIAPIPE_ASSERT_OK(
      output_stream_managers.Index(1).Initialize("output1", &output1_type));

  PacketSet input_side_packets(tool::CreateTagMap({}).ValueOrDie());

  CalculatorState calculator_state("Node", /*node_id=*/0, "Calculator",
                                   CalculatorGraphConfig::Node(), nullptr);

  calculator_state.SetInputSidePackets(&input_side_packets);

  CalculatorContextManager calculator_context_manager;
  CalculatorContext calculator_context(&calculator_state,
                                       tool::CreateTagMap({}).ValueOrDie(),
                                       output_stream_managers.TagMap());
  InputStreamShardSet& input_set = calculator_context.Inputs();
  OutputStreamShardSet& output_set = calculator_context.Outputs();
  output_set.Index(0).SetSpec(output_stream_managers.Index(0).Spec());
  output_set.Index(0).SetNextTimestampBound(Timestamp(10));
  output_set.Index(1).SetSpec(output_stream_managers.Index(1).Spec());
  output_set.Index(1).SetNextTimestampBound(Timestamp(11));
  CalculatorContextManager().PushInputTimestampToContext(
      &calculator_context, Timestamp::Unstarted());

  InputStreamSet input_streams(input_set.TagMap());
  OutputStreamSet output_streams(output_set.TagMap());
  for (CollectionItemId id = input_streams.BeginId();
       id < input_streams.EndId(); ++id) {
    input_streams.Get(id) = &input_set.Get(id);
  }
  for (CollectionItemId id = output_streams.BeginId();
       id < output_streams.EndId(); ++id) {
    output_streams.Get(id) = &output_set.Get(id);
  }
  calculator_state.SetInputStreamSet(&input_streams);
  calculator_state.SetOutputStreamSet(&output_streams);

  test_ns::DeadEndCalculator calculator;
  EXPECT_EQ(Timestamp(10), calculator.SourceProcessOrder(&calculator_context));
  output_set.Index(0).SetNextTimestampBound(Timestamp(100));
  EXPECT_EQ(Timestamp(11), calculator.SourceProcessOrder(&calculator_context));
}

// Tests registration of a calculator within a namespace.
// DeadEndCalculator is registered in namespace "mediapipe::test_ns".
TEST(CalculatorTest, CreateByName) {
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByName(  //
      "mediapipe.test_ns.DeadEndCalculator"));

  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByName(  //
      ".mediapipe.test_ns.DeadEndCalculator"));

  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "alpha", ".mediapipe.test_ns.DeadEndCalculator"));

  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "alpha", "mediapipe.test_ns.DeadEndCalculator"));

  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe", "mediapipe.test_ns.DeadEndCalculator"));

  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe.test_ns.sub_ns", "DeadEndCalculator"));

  EXPECT_EQ(CalculatorBaseRegistry::CreateByNameInNamespace(  //
                "mediapipe", "DeadEndCalculator")
                .status()
                .code(),
            ::mediapipe::StatusCode::kNotFound);

  EXPECT_EQ(CalculatorBaseRegistry::CreateByName(  //
                "DeadEndCalculator")
                .status()
                .code(),
            ::mediapipe::StatusCode::kNotFound);
}

// Tests registration of a calculator within a whitelisted namespace.
TEST(CalculatorTest, CreateByNameWhitelisted) {
  // Reset the registration namespace whitelist.
  *const_cast<std::unordered_set<std::string>*>(
      &NamespaceWhitelist::TopNamespaces()) = std::unordered_set<std::string>{
      "mediapipe::test_ns::whitelisted_ns",
      "mediapipe",
  };

  // Register a whitelisted calculator.
  CalculatorBaseRegistry::Register(
      "::mediapipe::test_ns::whitelisted_ns::DeadCalculator",
      absl::make_unique< ::mediapipe::test_ns::whitelisted_ns::DeadCalculator>);

  // A whitelisted calculator can be found in its own namespace.
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "", "mediapipe.test_ns.whitelisted_ns.DeadCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe.sub_ns", "test_ns.whitelisted_ns.DeadCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe.sub_ns", "mediapipe.EndCalculator"));

  // A whitelisted calculator can be found in the top-level namespace.
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "", "DeadCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe", "DeadCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe.test_ns.sub_ns", "DeadCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "", "EndCalculator"));
  MEDIAPIPE_EXPECT_OK(CalculatorBaseRegistry::CreateByNameInNamespace(  //
      "mediapipe.test_ns.sub_ns", "EndCalculator"));
}

}  // namespace
}  // namespace mediapipe