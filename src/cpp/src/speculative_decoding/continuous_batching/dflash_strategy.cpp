// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

namespace ov::genai {

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                   const ov::genai::ModelDesc& draft_model_desc) {
    (void)draft_model_desc;

    m_tokenizer = main_model_desc.tokenizer;

    auto main_model = main_model_desc.model;
    OPENVINO_ASSERT(main_model != nullptr, "Main model cannot be null for DFlashDecodingImpl");

    if (main_model_desc.inputs_embedder) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_vision_registry = std::make_shared<VisionRegistry>();
        m_main_cb_pipeline = std::make_shared<ContinuousBatchingImpl>(main_model,
                                                                       m_inputs_embedder,
                                                                       main_model_desc.tokenizer,
                                                                       main_model_desc.scheduler_config,
                                                                       main_model_desc.device,
                                                                       main_model_desc.properties,
                                                                       main_model_desc.generation_config,
                                                                       false);
    } else {
        m_main_cb_pipeline = std::make_shared<ContinuousBatchingImpl>(main_model,
                                                                       main_model_desc.tokenizer,
                                                                       main_model_desc.scheduler_config,
                                                                       main_model_desc.device,
                                                                       main_model_desc.properties,
                                                                       main_model_desc.generation_config,
                                                                       false);
    }

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
}

GenerationHandle
ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(uint64_t request_id,
                                                            const ov::Tensor& input_ids,
                                                            const ov::genai::GenerationConfig& sampling_params,
                                                            std::optional<ov::Tensor> token_type_ids,
                                                            std::optional<ov::Tensor> prompt_ids,
                                                            std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    return m_main_cb_pipeline->add_request(
        request_id, input_ids, sampling_params, token_type_ids, prompt_ids, lm_extra_inputs);
}

GenerationHandle
ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(uint64_t request_id,
                                                            const std::string& prompt,
                                                            const ov::genai::GenerationConfig& sampling_params) {
    return m_main_cb_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_non_finished_requests() {
    return m_main_cb_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    m_main_cb_pipeline->step();
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    return m_main_cb_pipeline->generate(
        input_ids, sampling_params, streamer, token_type_ids, position_ids, prompt_ids, lm_extra_inputs_list);
}

}  // namespace ov::genai
