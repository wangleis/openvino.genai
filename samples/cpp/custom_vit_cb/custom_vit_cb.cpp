// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/tensor.hpp"

std::string read_text_file(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open text file: " + file_path.string());
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

ov::Tensor read_binary_as_u8_tensor(const std::filesystem::path& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + file_path.string());
    }
    const std::streamoff file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Failed to get file size for: " + file_path.string());
    }
    file.seekg(0, std::ios::beg);

    ov::Tensor weights(ov::element::u8, {static_cast<size_t>(file_size)});
    file.read(reinterpret_cast<char*>(weights.data<uint8_t>()), file_size);
    if (!file) {
        throw std::runtime_error("Failed to read binary file: " + file_path.string());
    }
    return weights;
}

ov::Tensor load_tensor(const std::string& tensor_stem) {
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
    std::string meta_file = tensor_stem + "_meta.txt";
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
    std::string data_file = tensor_stem + "_data.txt";
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

ov::Tensor merge_deepstacks(const std::vector<ov::Tensor>& deepstack_embeds) {
    if (deepstack_embeds.empty()) {
        throw std::runtime_error("No deepstack embeds provided for merging.");
    }
    const auto& first_shape = deepstack_embeds[0].get_shape();
    if (first_shape.size() != 3) {
        throw std::runtime_error("Expected deepstack embeds to have rank 3, but got rank " +
                                 std::to_string(first_shape.size()));
    }
    size_t total_layers = 0;
    size_t max_tokens = 0;
    size_t hidden_size = first_shape[2];
    for (const auto& embed : deepstack_embeds) {
        const auto& shape = embed.get_shape();
        if (shape.size() != 3 || shape[2] != hidden_size) {
            throw std::runtime_error(
                "Inconsistent deepstack embed shapes. All embeds must have the same hidden size and rank 3.");
        }
        total_layers += shape[0];
        max_tokens = std::max(max_tokens, shape[1]);
    }

    ov::Tensor merged(ov::element::f32, {total_layers, max_tokens, hidden_size});
    float* merged_data = merged.data<float>();
    size_t layer_offset = 0;
    for (const auto& embed : deepstack_embeds) {
        const auto& shape = embed.get_shape();
        size_t num_layers = shape[0];
        size_t num_tokens = shape[1];
        const float* embed_data = embed.data<float>();
        for (size_t layer = 0; layer < num_layers; ++layer) {
            for (size_t token = 0; token < num_tokens; ++token) {
                size_t src_idx = layer * num_tokens * hidden_size + token * hidden_size;
                size_t dst_idx = (layer_offset + layer) * max_tokens * hidden_size + token * hidden_size;
                std::copy(embed_data + src_idx, embed_data + src_idx + hidden_size, merged_data + dst_idx);
            }
        }
        layer_offset += num_layers;
    }
    return merged;
}

ov::Tensor compact_deepstack_with_visual_pos_mask(const ov::Tensor& fullseq_deepstack_embeds,
                                                  const ov::Tensor& visual_pos_mask) {
    const ov::Shape deepstack_shape = fullseq_deepstack_embeds.get_shape();
    if (deepstack_shape.size() != 3) {
        throw std::runtime_error("Expected fullseq_deepstack_embeds rank 3.");
    }
    const size_t num_layers = deepstack_shape[0];
    const size_t seq_len = deepstack_shape[1];
    const size_t hidden_size = deepstack_shape[2];

    if (visual_pos_mask.get_element_type() != ov::element::boolean) {
        throw std::runtime_error("Expected visual_pos_mask element type bool.");
    }
    if (visual_pos_mask.get_size() != seq_len) {
        throw std::runtime_error("visual_pos_mask size must match deepstack sequence length.");
    }

    const bool* mask_data = visual_pos_mask.data<const bool>();
    size_t vision_tokens = 0;
    for (size_t i = 0; i < seq_len; ++i) {
        if (mask_data[i]) {
            vision_tokens++;
        }
    }

    // Keep second dim >= 1 for no-vision-token edge case.
    const size_t compact_tokens = std::max<size_t>(vision_tokens, 1);
    ov::Tensor compact_deepstack(ov::element::f32, {num_layers, compact_tokens, hidden_size});
    std::fill_n(compact_deepstack.data<float>(), compact_deepstack.get_size(), 0.0f);

    const float* src = fullseq_deepstack_embeds.data<const float>();
    float* dst = compact_deepstack.data<float>();

    for (size_t layer = 0; layer < num_layers; ++layer) {
        size_t compact_idx = 0;
        for (size_t pos = 0; pos < seq_len; ++pos) {
            if (!mask_data[pos]) {
                continue;
            }
            const size_t src_offset = (layer * seq_len + pos) * hidden_size;
            const size_t dst_offset = (layer * compact_tokens + compact_idx) * hidden_size;
            std::copy_n(src + src_offset, hidden_size, dst + dst_offset);
            compact_idx++;
        }
    }

    return compact_deepstack;
}

size_t get_latest_step(const std::string& data_dir, const std::string& prefix, const size_t default_step = static_cast<size_t>(-1)) {
    const std::regex pattern("^" + prefix + "_(\\d+)_meta\\.txt$");
    bool has_default_step = false;
    std::optional<size_t> latest_step = std::nullopt;

    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(filename, match, pattern)) {
            continue;
        }
        const size_t step = static_cast<size_t>(std::stoull(match[1].str()));
        if (step == default_step) {
            has_default_step = true;
        }
        if (!latest_step.has_value() || step > *latest_step) {
            latest_step = step;
        }
    }

    if (has_default_step) {
        return default_step;
    }

    if (!latest_step.has_value()) {
        throw std::runtime_error("Cannot find tensor step for prefix: " + prefix);
    }
    return *latest_step;
}

int main(int argc, char* argv[]) {
    // Hardcode input parameters for debugging.
    const std::filesystem::path model_dir =
        "/home/xiping/mygithub/modular_genai/composable_pipeline/tests/test_models/Qwen3-Omni-4B-Instruct-multilingual-int4";
    const std::string device = "CPU";

    std::string data_dir =
        "/home/xiping/mygithub/modular_genai/composable_pipeline/tests/cpp/test_data/llm_inputs_data/";

    const size_t selected_step = get_latest_step(data_dir, "inputs_embeds", 0);
    std::cout << "Loading tensors for step: " << selected_step << std::endl;

    ov::Tensor inputs_embeds = load_tensor(data_dir + "inputs_embeds_" + std::to_string(selected_step));
    ov::Tensor visual_pos_mask = load_tensor(data_dir + "visual_pos_mask_" + std::to_string(selected_step));
    ov::Tensor deepstack_embeds_0 = load_tensor(data_dir + "deepstack_embeds_0_" + std::to_string(selected_step));
    ov::Tensor deepstack_embeds_1 = load_tensor(data_dir + "deepstack_embeds_1_" + std::to_string(selected_step));
    ov::Tensor deepstack_embeds_2 = load_tensor(data_dir + "deepstack_embeds_2_" + std::to_string(selected_step));
    ov::Tensor position_ids = load_tensor(data_dir + "position_ids_" + std::to_string(selected_step));
    ov::Tensor merged_deepstack_embeds = merge_deepstacks({deepstack_embeds_0, deepstack_embeds_1, deepstack_embeds_2});
    ov::Tensor compact_deepstack_embeds = compact_deepstack_with_visual_pos_mask(merged_deepstack_embeds, visual_pos_mask);

    std::unordered_map<std::string, ov::Tensor> extra_inputs;
    extra_inputs["visual_pos_masks"] = visual_pos_mask;
    extra_inputs["deepstack_visual_embeds"] = compact_deepstack_embeds;
    std::cout << "  merged_deepstack_embeds shape: " << merged_deepstack_embeds.get_shape() << std::endl;
    std::cout << "  compact_deepstack_embeds shape: " << compact_deepstack_embeds.get_shape() << std::endl;
    std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>> extra_inputs_list = std::nullopt;
    if (!extra_inputs.empty()) {
        extra_inputs_list = std::vector<std::unordered_map<std::string, ov::Tensor>>{std::move(extra_inputs)};
    }

    std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>> position_ids_list =
        std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>{{position_ids, std::nullopt}};

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = 1;

    ov::genai::ModelsMap models_map;
    models_map["language"] = {
        read_text_file(model_dir / "openvino_language_model.xml"),
        read_binary_as_u8_tensor(model_dir / "openvino_language_model.bin")
    };
    models_map["text_embeddings"] = {
        read_text_file(model_dir / "openvino_text_embeddings_model.xml"),
        read_binary_as_u8_tensor(model_dir / "openvino_text_embeddings_model.bin")
    };
    models_map["vision_embeddings"] = {
        std::string(),
        ov::Tensor()};  // dummy vision embed model to trigger modeling VL code path in InputsEmbedder.
    const ov::genai::Tokenizer tokenizer(model_dir);

    const ov::AnyMap props;
    auto m_pipeline = std::make_unique<ov::genai::ContinuousBatchingPipeline>(
        models_map,
        tokenizer,
        scheduler_config,
        device,
        model_dir,
        props);

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
        text = m_pipeline->get_tokenizer().decode(results[0].m_generation_ids[0]);
    }
    std::cout << "Generated text: " << text << std::endl;

    std::string expected_text = R"(**Summary:** The forecast indicates a transition from sunny conditions to a stormy day, with a high chance of thunderstorms. The visual evidence of a heavy downpour and the sound of thunder confirm that the weather is indeed stormy, and the forecast is accurate.

        * *Voice Alert : **The forecast is correct.A storm )";
    return 0;
}