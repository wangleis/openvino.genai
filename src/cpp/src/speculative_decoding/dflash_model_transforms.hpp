// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

/**
 * @brief Runtime configuration for DFlash speculative decoding.
 */
struct DFlashRTInfo {
    bool dflash_mode = false;                            ///< Enable DFlash mode
    std::vector<int32_t> hidden_layers_to_abstract;     ///< Layer indices used for hidden state abstraction
};

/**
 * @brief Extracts DFlash configuration from model config.
 * @param config Model configuration map.
 * @param models_path Reserved for parity with Eagle3 extractor and future defaults.
 * @return DFlashRTInfo structure with extracted configuration.
 * @note If dflash_mode is enabled and hidden_layers_list is not provided, layer ids are derived from
 *       config.json using Torch-style build_target_layer_ids behavior.
 */
DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});

/**
 * @brief Extracts hidden states from specified decoder layers for DFlash.
 *
 * Mirrors Eagle3 hidden-state extraction behavior and appends a unified
 * `last_hidden_state` result required by DFlash direct block decoding.
 *
 * @param model Model to transform.
 * @param hidden_layers_to_abstract Layer indices to extract.
 */
void transform_hidden_state(std::shared_ptr<ov::Model>& model,
                            const std::vector<int32_t>& hidden_layers_to_abstract);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
