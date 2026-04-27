// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_draft_strategy.hpp"

#include <limits>
#include <vector>

namespace ov::genai {
class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
protected:
    struct DFlashPendingRequest {
        ov::Tensor input_ids;
        ov::genai::GenerationConfig sampling_params;
        std::optional<ov::Tensor> token_type_ids;
        std::optional<ov::Tensor> prompt_ids;
    };

    std::shared_ptr<ov::Model> m_main_model;
    std::shared_ptr<ov::Model> m_main_direct_model;
    std::shared_ptr<ov::Model> m_draft_model;
    ov::InferRequest m_main_request;
    ov::InferRequest m_draft_request;
    ov::genai::utils::KVAxesPosition m_main_kv_axes_pos;
    std::string m_main_device;
    ov::AnyMap m_main_properties;
    ov::genai::SchedulerConfig m_main_scheduler_config;
    ov::genai::GenerationConfig m_main_generation_config;
    bool m_direct_dflash_ready = false;
    bool m_draft_speculative_pipeline_ready = false;
    std::string m_main_logits_output_name;
    std::string m_main_hidden_output_name;
    std::string m_draft_logits_output_name;
    size_t m_main_logits_output_index = 0;
    size_t m_main_hidden_output_index = std::numeric_limits<size_t>::max();
    size_t m_draft_logits_output_index = 0;
    std::vector<int32_t> m_hidden_layers_to_abstract;
    std::map<uint64_t, DFlashPendingRequest> m_dflash_pending_requests;

    bool should_use_direct_block_decode(const std::vector<ov::Tensor>& input_ids,
                                        const std::vector<GenerationConfig>& sampling_params,
                                        const StreamerVariant& streamer,
                                        const std::optional<std::vector<ov::Tensor>>& token_type_ids,
                                        const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
                                        const std::optional<std::vector<ov::Tensor>>& prompt_ids,
                                        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) const;

    EncodedGenerationResult direct_block_decode_generate(const ov::Tensor& input_ids,
                                                         const GenerationConfig& sampling_params,
                                                         const StreamerVariant& streamer,
                                                         const std::optional<ov::Tensor>& token_type_ids,
                                                         const std::optional<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids,
                                                         const std::optional<ov::Tensor>& prompt_ids,
                                                         const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs);

    ov::Tensor run_main_model_step(const ov::Tensor& token_ids,
                                   size_t start_pos,
                                   const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs,
                                   std::optional<ov::Tensor>& hidden_state,
                                   bool apply_extra_inputs,
                                   const std::optional<ov::Tensor>& input_embeds = std::nullopt);

    ov::Tensor run_main_model_prefill_with_embeddings(
        const ov::Tensor& inputs_embeds,
        const std::optional<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids,
        const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs,
        std::optional<ov::Tensor>& hidden_state);

    ov::Tensor run_draft_model_step(const ov::Tensor& noise_embedding,
                                    const ov::Tensor& target_hidden,
                                    size_t block_size);

    void trim_main_kv_cache(size_t tokens_to_trim, size_t current_sequence_len);

    static int64_t greedy_pick_last_token(const ov::Tensor& logits, size_t position_idx);
    static ov::Tensor build_i64_arange_row(size_t start, size_t length);
    static ov::Tensor build_position_ids_like(const ov::Output<ov::Node>& position_input, size_t start, size_t length);
    static ov::Tensor build_attention_mask_like(const ov::Output<ov::Node>& mask_input, size_t length);
    static ov::Tensor slice_hidden_prefix(const ov::Tensor& hidden_state, size_t prefix_len);

public:
    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                       const ov::genai::ModelDesc& draft_model_desc,
                       const std::vector<int32_t>& hidden_layers_to_abstract);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;
};
}  // namespace ov::genai
