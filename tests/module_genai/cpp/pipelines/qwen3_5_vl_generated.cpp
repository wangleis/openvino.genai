// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"

using test_params = std::tuple<std::string>;
using namespace ov::genai::module;

class Qwen3_5_VL_Generated_Pipeline_Test : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        return std::string("GeneratedPipeline_") + std::get<0>(obj.param);
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_device) = GetParam();
    }

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen3_5_vl";
        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node pipeline_params;
        pipeline_params["type"] = "ParameterModule";
        pipeline_params["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        pipeline_params["outputs"].push_back(output_node("img1", "OVTensor"));
        pipeline_params["outputs"].push_back(output_node("prompts_data", "String"));
        pipeline_modules["pipeline_params"] = pipeline_params;

        YAML::Node image_preprocessor;
        image_preprocessor["type"] = "GeneratedImagePreprocessorModule";
        image_preprocessor["device"] = m_device;
        image_preprocessor["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        image_preprocessor["inputs"].push_back(input_node("image", "OVTensor", "pipeline_params.img1"));
        image_preprocessor["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        image_preprocessor["outputs"].push_back(output_node("raw_data", "OVTensor"));
        image_preprocessor["outputs"].push_back(output_node("source_size", "VecInt"));
        image_preprocessor["params"] = YAML::Node();
        image_preprocessor["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        pipeline_modules["image_preprocessor"] = image_preprocessor;

        YAML::Node prompt_encoder;
        prompt_encoder["type"] = "GeneratedPromptEncoderModule";
        prompt_encoder["device"] = m_device;
        prompt_encoder["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        prompt_encoder["inputs"].push_back(input_node("prompts", "String", "pipeline_params.prompts_data"));
        prompt_encoder["inputs"].push_back(input_node("encoded_image", "OVTensor", "image_preprocessor.raw_data"));
        prompt_encoder["inputs"].push_back(input_node("source_size", "VecInt", "image_preprocessor.source_size"));
        prompt_encoder["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        prompt_encoder["outputs"].push_back(output_node("input_ids", "OVTensor"));
        prompt_encoder["outputs"].push_back(output_node("mask", "OVTensor"));
        prompt_encoder["outputs"].push_back(output_node("images_sequence", "VecInt"));
        prompt_encoder["params"] = YAML::Node();
        prompt_encoder["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        pipeline_modules["prompt_encoder"] = prompt_encoder;

        YAML::Node text_embedding;
        text_embedding["type"] = "GeneratedTextEmbeddingModule";
        text_embedding["device"] = m_device;
        text_embedding["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        text_embedding["inputs"].push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
        text_embedding["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        text_embedding["outputs"].push_back(output_node("input_embedding", "OVTensor"));
        text_embedding["params"] = YAML::Node();
        text_embedding["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        text_embedding["params"]["scale_emb"] = "1.0";
        pipeline_modules["text_embedding"] = text_embedding;

        YAML::Node vision_encoder;
        vision_encoder["type"] = "GeneratedVisionEncoderModule";
        vision_encoder["device"] = m_device;
        vision_encoder["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        vision_encoder["inputs"].push_back(input_node("preprocessed_image", "OVTensor", "image_preprocessor.raw_data"));
        vision_encoder["inputs"].push_back(input_node("source_size", "VecInt", "image_preprocessor.source_size"));
        vision_encoder["inputs"].push_back(input_node("images_sequence", "VecInt", "prompt_encoder.images_sequence"));
        vision_encoder["inputs"].push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
        vision_encoder["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        vision_encoder["outputs"].push_back(output_node("image_embedding", "OVTensor"));
        vision_encoder["outputs"].push_back(output_node("video_embedding", "OVTensor"));
        vision_encoder["outputs"].push_back(output_node("position_ids", "OVTensor"));
        vision_encoder["outputs"].push_back(output_node("rope_delta", "Int"));
        vision_encoder["params"] = YAML::Node();
        vision_encoder["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        vision_encoder["params"]["vision_start_token_id"] = "151652";
        pipeline_modules["vision_encoder"] = vision_encoder;

        YAML::Node embedding_merger;
        embedding_merger["type"] = "GeneratedEmbeddingMergerModule";
        embedding_merger["device"] = m_device;
        embedding_merger["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        embedding_merger["inputs"].push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
        embedding_merger["inputs"].push_back(input_node("input_embedding", "OVTensor", "text_embedding.input_embedding"));
        embedding_merger["inputs"].push_back(input_node("image_embedding", "OVTensor", "vision_encoder.image_embedding"));
        embedding_merger["inputs"].push_back(input_node("video_embedding", "OVTensor", "vision_encoder.video_embedding"));
        embedding_merger["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        embedding_merger["outputs"].push_back(output_node("merged_embedding", "OVTensor"));
        embedding_merger["params"] = YAML::Node();
        embedding_merger["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        pipeline_modules["embedding_merger"] = embedding_merger;

        YAML::Node pipeline_results;
        pipeline_results["type"] = "ResultModule";
        pipeline_results["device"] = "CPU";
        pipeline_results["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        pipeline_results["inputs"].push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
        pipeline_results["inputs"].push_back(input_node("merged_embedding", "OVTensor", "embedding_merger.merged_embedding"));
        pipeline_results["inputs"].push_back(input_node("position_ids", "OVTensor", "vision_encoder.position_ids"));
        pipeline_modules["pipeline_results"] = pipeline_results;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["prompts_data"] = std::vector<std::string>{"Please describe this image"};
        inputs["img1"] = utils::load_image(TEST_DATA::img_cat_120_100());
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto input_ids = pipe.get_output("input_ids").as<ov::Tensor>();
        auto merged_embedding = pipe.get_output("merged_embedding").as<ov::Tensor>();
        auto position_ids = pipe.get_output("position_ids").as<ov::Tensor>();

        EXPECT_GT(input_ids.get_size(), 0u);
        EXPECT_EQ(input_ids.get_shape().size(), 2u);

        EXPECT_GT(merged_embedding.get_size(), 0u);
        EXPECT_EQ(merged_embedding.get_shape().size(), 3u);

        EXPECT_GT(position_ids.get_size(), 0u);
        EXPECT_EQ(position_ids.get_shape().size(), 3u);
    }
};

TEST_P(Qwen3_5_VL_Generated_Pipeline_Test, PipelineTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         Qwen3_5_VL_Generated_Pipeline_Test,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         Qwen3_5_VL_Generated_Pipeline_Test::get_test_case_name);
