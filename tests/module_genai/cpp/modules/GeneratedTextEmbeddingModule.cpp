// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"

using test_params = std::tuple<std::string>;
using namespace ov::genai::module;

class GeneratedTextEmbeddingModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    float m_threshold = 1e-5f;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<0>(obj.param);
        return std::string("GeneratedTextEmbedding_") + device;
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

        YAML::Node text_embedding;
        text_embedding["type"] = "GeneratedTextEmbeddingModule";
        text_embedding["device"] = m_device;
        text_embedding["inputs"] = YAML::Node(YAML::NodeType::Sequence);
        text_embedding["inputs"].push_back(
            input_node("input_ids", to_string(DataType::OVTensor), "pipeline_params.input_ids"));
        text_embedding["outputs"] = YAML::Node(YAML::NodeType::Sequence);
        text_embedding["outputs"].push_back(output_node("input_embedding", to_string(DataType::OVTensor)));
        text_embedding["params"] = YAML::Node();
        text_embedding["params"]["model_path"] = TEST_MODEL::Qwen2_5_VL_3B_Instruct_INT4();
        text_embedding["params"]["scale_emb"] = "1.0";
        pipeline_modules["text_embedding"] = text_embedding;

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto input_ids = ov::Tensor(ov::element::i64, ov::Shape{1, 6});
        int64_t* data_ptr = input_ids.data<int64_t>();
        std::vector<int64_t> values = {1986, 374, 264, 6077, 9934, 13};
        std::copy(values.begin(), values.end(), data_ptr);

        inputs["input_ids"] = input_ids;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("input_embedding").as<ov::Tensor>();
        const std::vector<float> expected_text_embeds = {
            0.0129318f, 0.000862122f, 0.0021553f, 0.0f, -0.0133667f, 0.0168152f, 0.00387955f, 0.0021553f, -0.0375061f, -0.0241394f};

        EXPECT_TRUE(compare_big_tensor(output, expected_text_embeds, m_threshold))
            << "input_embedding does not match expected values within threshold " << m_threshold;
        EXPECT_TRUE(compare_shape(output.get_shape(), ov::Shape{1, 6, 2048}))
            << "input_embedding shape does not match expected shape";
    }
};

TEST_P(GeneratedTextEmbeddingModuleTest, ModuleTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         GeneratedTextEmbeddingModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices)),
                         GeneratedTextEmbeddingModuleTest::get_test_case_name);
