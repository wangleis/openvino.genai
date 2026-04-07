// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/runtime/tensor.hpp"

int main(int argc, char* argv[]) {
    // Hardcode input parameters for debugging.
    const std::filesystem::path model_dir = "/home/xiping/mygithub/modular_genai/composable_pipeline/tests/test_models/"
                                            "Qwen3-Omni-4B-Instruct-multilingual-int4";
    const std::string device = "CPU";

    std::string data_dir =
        "/home/xiping/mygithub/modular_genai/composable_pipeline/tests/cpp/test_data/llm_inputs_data/";
    auto load_tensor = [&](const std::string& filename) -> ov::Tensor {
        auto trim = [](std::string value) {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                            return !std::isspace(ch);
                        }));
            value.erase(std::find_if(value.rbegin(),
                                     value.rend(),
                                     [](unsigned char ch) {
                                         return !std::isspace(ch);
                                     })
                            .base(),
                        value.end());
            return value;
        };
        // load shape and type.
        std::string meta_file = data_dir + filename + "_0_meta.txt";
        // Reference meta file content:
        // shape: 1 1 2560
        // element_type: f32
        std::ifstream meta_ifs(meta_file);
        if (!meta_ifs.is_open()) {
            throw std::runtime_error("Failed to open meta file: " + meta_file);
        }
        std::string line;
        ov::Shape shape;
        ov::element::Type type;
        while (std::getline(meta_ifs, line)) {
            if (line.find("shape:") == 0) {
                std::istringstream iss(line.substr(6));
                size_t dim;
                while (iss >> dim) {
                    shape.push_back(dim);
                }
            } else if (line.find("element_type:") == 0) {
                std::string type_str = trim(line.substr(13));
                type = ov::element::Type(type_str);
            }
        }

        // based shape size and type, load data.
        ov::Tensor tensor(type, shape);
        std::string data_file = data_dir + filename + "_0_data.txt";
        std::ifstream data_ifs(data_file);
        if (!data_ifs.is_open()) {
            throw std::runtime_error("Failed to open data file: " + data_file);
        }
        if (type == ov::element::f32) {
            auto* data_ptr = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::i64) {
            auto* data_ptr = tensor.data<int64_t>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::i32) {
            auto* data_ptr = tensor.data<int32_t>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::boolean) {
            auto* data_ptr = tensor.data<bool>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                int val;
                data_ifs >> val;
                data_ptr[i] = static_cast<bool>(val);
            }
        } else {
            throw std::runtime_error("Unsupported element type in meta file: " + type.to_string());
        }
        return tensor;
    };

    ov::Tensor inputs_embeds = load_tensor("inputs_embeds");
    ov::Tensor attention_mask = load_tensor("attention_mask");
    ov::Tensor beam_idx = load_tensor("beam_idx");
    ov::Tensor visual_pos_mask = load_tensor("visual_pos_mask");
    ov::Tensor deepstack_embeds_0 = load_tensor("deepstack_embeds_0");
    ov::Tensor deepstack_embeds_1 = load_tensor("deepstack_embeds_1");
    ov::Tensor deepstack_embeds_2 = load_tensor("deepstack_embeds_2");
    ov::Tensor position_ids = load_tensor("position_ids");

    std::unordered_map<std::string, ov::Tensor> extra_inputs;
    extra_inputs["attention_mask"] = attention_mask;
    extra_inputs["beam_idx"] = beam_idx;
    extra_inputs["visual_pos_mask"] = visual_pos_mask;
    std::vector<ov::Tensor> deepstack_embeds = {deepstack_embeds_0, deepstack_embeds_1, deepstack_embeds_2};
    for (size_t i = 0; i < deepstack_embeds.size(); ++i) {
        extra_inputs["deepstack_embeds." + std::to_string(i)] = deepstack_embeds.at(i);
    }
    std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>> extra_inputs_list = std::nullopt;
    if (!extra_inputs.empty()) {
        extra_inputs_list = std::vector<std::unordered_map<std::string, ov::Tensor>>{std::move(extra_inputs)};
    }

    std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>> position_ids_list =
        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>{{position_ids, std::nullopt}};

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();

    const ov::AnyMap props;
    auto m_pipeline =
        std::make_unique<ov::genai::ContinuousBatchingPipeline>(model_dir, scheduler_config, device, props);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 64;

    const auto results = m_pipeline->generate({inputs_embeds},
                                              {generation_config},
                                              std::monostate(),
                                              std::nullopt,
                                              position_ids_list,
                                              extra_inputs_list);

    std::string text;
    if (!results.empty() && !results[0].m_generation_ids.empty() && !results[0].m_generation_ids[0].empty()) {
        ov::genai::Tokenizer tokenizer = m_pipeline->get_tokenizer();
        text = tokenizer.decode(results[0].m_generation_ids[0]);
    }
    std::cout << "Generated text: " << text << std::endl;

    std::string expected_text = R"(**Summary:** The forecast indicates a transition from sunny conditions to a stormy day, with a high chance of thunderstorms. The visual evidence of a heavy downpour and the sound of thunder confirm that the weather is indeed stormy, and the forecast is accurate.

        * *Voice Alert : **The forecast is correct.A storm )";
    return 0;
}