// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "eagle3_model_transforms.hpp"
#include "json_utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

std::vector<int32_t> build_target_layer_ids_from_num_layers(int32_t num_hidden_layers) {
    // Torch build_target_layer_ids equivalent: 5 evenly spaced ids from [1, num_hidden_layers - 3].
    // For num_hidden_layers=36 this yields [1, 9, 17, 25, 33].
    constexpr int32_t NUM_TARGET_LAYERS = 5;
    OPENVINO_ASSERT(num_hidden_layers >= 5,
                    "num_hidden_layers must be >= 5 for DFlash target layer id construction, got: ",
                    num_hidden_layers);

    const int32_t start = 1;
    const int32_t end = num_hidden_layers - 3;
    const int32_t step = (end - start) / (NUM_TARGET_LAYERS - 1);

    std::vector<int32_t> layer_ids;
    layer_ids.reserve(NUM_TARGET_LAYERS);
    for (int32_t i = 0; i < NUM_TARGET_LAYERS; ++i) {
        layer_ids.push_back(start + i * step);
    }
    return layer_ids;
}

std::filesystem::path resolve_config_json_path(const std::filesystem::path& models_path) {
    if (models_path.empty()) {
        return {};
    }

    if (std::filesystem::is_directory(models_path)) {
        return models_path / "config.json";
    }

    return models_path.parent_path() / "config.json";
}

}  // namespace

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    DFlashRTInfo dflash_rt_info;

    auto mode_it = config.find("dflash_mode");
    if (mode_it == config.end()) {
        return dflash_rt_info;
    }

    dflash_rt_info.dflash_mode = mode_it->second.as<bool>();
    config.erase(mode_it);

    if (!dflash_rt_info.dflash_mode) {
        return dflash_rt_info;
    }

    auto layers_it = config.find("hidden_layers_list");
    if (layers_it != config.end()) {
        OPENVINO_ASSERT(layers_it->second.is<std::vector<int32_t>>(),
                        "hidden_layers_list must be a vector of int32_t values");
        dflash_rt_info.hidden_layers_to_abstract = layers_it->second.as<std::vector<int32_t>>();
        config.erase(layers_it);
    }

    if (dflash_rt_info.hidden_layers_to_abstract.empty()) {
        const auto config_file_path = resolve_config_json_path(models_path);
        OPENVINO_ASSERT(!config_file_path.empty() && std::filesystem::exists(config_file_path),
                        "Cannot deduce DFlash hidden layers because config.json is missing: ",
                        config_file_path.empty() ? std::string("<empty path>") : config_file_path.string());

        std::ifstream file(config_file_path);
        nlohmann::json data = nlohmann::json::parse(file);

        int32_t num_hidden_layers = 0;
        using ov::genai::utils::read_json_param;
        read_json_param(data, "num_hidden_layers", num_hidden_layers);
        if (num_hidden_layers == 0) {
            read_json_param(data, "text_config.num_hidden_layers", num_hidden_layers);
            if (num_hidden_layers == 0) {
                read_json_param(data, "thinker_config.text_config.num_hidden_layers", num_hidden_layers);
            }
        }

        OPENVINO_ASSERT(num_hidden_layers > 0,
                        "Failed to read num_hidden_layers from config.json for DFlash hidden layer construction: ",
                        config_file_path.string());

        dflash_rt_info.hidden_layers_to_abstract = build_target_layer_ids_from_num_layers(num_hidden_layers);
    }

    return dflash_rt_info;
}

void transform_hidden_state(std::shared_ptr<ov::Model>& model,
                            const std::vector<int32_t>& hidden_layers_to_abstract) {
    OPENVINO_ASSERT(model, "DFlash transform_hidden_state received null model");
    OPENVINO_ASSERT(!hidden_layers_to_abstract.empty(),
                    "DFlash transform_hidden_state requires non-empty hidden_layers_to_abstract");

    // Reuse the validated residual-node extraction logic from Eagle3 transforms.
    ov::genai::utils::eagle3::transform_hidden_state(model, hidden_layers_to_abstract);
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
