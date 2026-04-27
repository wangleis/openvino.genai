// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "openvino/op/transpose.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace {

template <typename NodeT>
bool port_matches_name(const ov::Output<NodeT>& port, const std::string& name) {
    if (port.get_names().count(name) > 0) {
        return true;
    }

    const auto node = port.get_node_shared_ptr();
    if (node) {
        if (node->get_friendly_name() == name) {
            return true;
        }
        if (node->get_name() == name) {
            return true;
        }
    }

    return false;
}

template <typename NodeT>
bool has_port_named(const std::vector<ov::Output<NodeT>>& ports, const std::string& name) {
    for (const auto& port : ports) {
        if (port_matches_name(port, name)) {
            return true;
        }
    }
    return false;
}

template <typename NodeT>
std::optional<ov::Output<NodeT>> find_port_by_name(const std::vector<ov::Output<NodeT>>& ports,
                                                   const std::string& name) {
    for (const auto& port : ports) {
        if (port_matches_name(port, name)) {
            return port;
        }
    }
    return std::nullopt;
}

template <typename NodeT>
std::optional<size_t> find_port_index_by_name(const std::vector<ov::Output<NodeT>>& ports,
                                              const std::string& name) {
    for (size_t idx = 0; idx < ports.size(); ++idx) {
        if (port_matches_name(ports[idx], name)) {
            return idx;
        }
    }
    return std::nullopt;
}

template <typename NodeT>
std::string first_tensor_name_or_empty(const ov::Output<NodeT>& port) {
    const auto& names = port.get_names();
    if (names.empty()) {
        return "";
    }
    return *names.begin();
}

template <typename NodeT>
std::string describe_port(const ov::Output<NodeT>& port) {
    std::ostringstream oss;
    oss << "friendly=" << port.get_node_shared_ptr()->get_friendly_name();
    oss << " pshape=" << port.get_partial_shape();
    oss << " tensor_names={";
    bool first = true;
    for (const auto& n : port.get_names()) {
        if (!first) {
            oss << ",";
        }
        oss << n;
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string shape_to_string(const ov::Shape& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void remove_deepstack_transpose(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return;
    }

    bool updated = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(node);
        if (!transpose) {
            continue;
        }
        const std::string name = transpose->get_friendly_name();
        if (name.rfind("deepstack_aligned_", 0) != 0) {
            continue;
        }
        const ov::Output<ov::Node> source = transpose->input_value(0);
        for (auto& output : transpose->outputs()) {
            for (auto target_input : output.get_target_inputs()) {
                target_input.replace_source_output(source);
            }
        }
        updated = true;
    }

    if (updated) {
        model->validate_nodes_and_infer_types();
    }
}

ov::Tensor adapt_position_ids_to_input(const ov::Tensor& position_ids, const ov::Output<ov::Node>& position_input) {
    const auto input_pshape = position_input.get_partial_shape();
    const auto src_shape = position_ids.get_shape();
    const bool input_rank_is_3 = input_pshape.rank().is_static() && input_pshape.rank().get_length() == 3;
    const bool input_rank_is_2 = input_pshape.rank().is_static() && input_pshape.rank().get_length() == 2;

    if (src_shape.size() == 3 && src_shape[1] == 1 && !input_rank_is_3) {
        ov::Tensor collapsed(ov::element::i64, {1, src_shape[2]});
        const int64_t* src = position_ids.data<const int64_t>();
        int64_t* dst = collapsed.data<int64_t>();
        std::copy_n(src, src_shape[2], dst);
        return collapsed;
    }

    if (input_rank_is_2) {
        if (src_shape.size() == 2) {
            return position_ids;
        }
        if (src_shape.size() == 3 && src_shape[1] == 1) {
            ov::Tensor collapsed(ov::element::i64, {1, src_shape[2]});
            const int64_t* src = position_ids.data<const int64_t>();
            int64_t* dst = collapsed.data<int64_t>();
            std::copy_n(src, src_shape[2], dst);
            return collapsed;
        }
    }

    if (input_rank_is_3) {
        if (src_shape.size() == 3) {
            return position_ids;
        }
        if (src_shape.size() == 2 && src_shape[0] == 1) {
            size_t rows = 3;
            if (input_pshape[0].is_static()) {
                rows = static_cast<size_t>(input_pshape[0].get_length());
            }
            ov::Tensor expanded(ov::element::i64, {rows, src_shape[1], 1});
            const int64_t* src = position_ids.data<const int64_t>();
            int64_t* dst = expanded.data<int64_t>();
            for (size_t r = 0; r < rows; ++r) {
                std::copy_n(src, src_shape[1], dst + r * src_shape[1]);
            }
            return expanded;
        }
    }

    return position_ids;
}

ov::Tensor normalize_visual_pos_mask_tensor(const ov::Tensor& visual_mask) {
    const auto shape = visual_mask.get_shape();
    if (shape.size() == 2) {
        return visual_mask;
    }
    if (shape.size() == 1) {
        ov::Tensor expanded(visual_mask.get_element_type(), {1, shape[0]});
        std::memcpy(expanded.data(), visual_mask.data(), visual_mask.get_byte_size());
        return expanded;
    }
    OPENVINO_THROW("visual_pos_mask rank ", shape.size(), " is not supported in DFlash direct mode");
}

ov::Tensor to_token_major_inputs_embeds(const ov::Tensor& inputs_embeds) {
    const auto shape = inputs_embeds.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "inputs_embeds must have rank 3");
    if (shape[0] != 1) {
        return inputs_embeds;
    }

    const size_t seq_len = shape[1];
    const size_t hidden = shape[2];
    ov::Tensor token_major(inputs_embeds.get_element_type(), {seq_len, 1, hidden});
    const size_t bytes_per_token = hidden * inputs_embeds.get_element_type().size();
    for (size_t s = 0; s < seq_len; ++s) {
        const char* src = static_cast<const char*>(inputs_embeds.data()) + s * bytes_per_token;
        char* dst = static_cast<char*>(token_major.data()) + s * bytes_per_token;
        std::memcpy(dst, src, bytes_per_token);
    }
    return token_major;
}

ov::Tensor to_token_major_visual_pos_mask(const ov::Tensor& visual_pos_mask_2d) {
    const auto shape = visual_pos_mask_2d.get_shape();
    OPENVINO_ASSERT(shape.size() == 2, "visual_pos_mask must be rank 2 before token-major conversion");
    if (shape[0] != 1) {
        return visual_pos_mask_2d;
    }

    ov::Tensor token_major(visual_pos_mask_2d.get_element_type(), {shape[1], 1});
    const bool* src = visual_pos_mask_2d.data<const bool>();
    bool* dst = token_major.data<bool>();
    for (size_t s = 0; s < shape[1]; ++s) {
        dst[s] = src[s];
    }
    return token_major;
}

ov::Tensor build_token_major_attention_mask_like(const ov::Output<ov::Node>& mask_input, size_t seq_len) {
    const auto mask_type = mask_input.get_element_type();
    ov::Tensor mask(mask_type, {seq_len, seq_len});

    if (mask_type == ov::element::boolean) {
        bool* ptr = mask.data<bool>();
        for (size_t r = 0; r < seq_len; ++r) {
            for (size_t c = 0; c < seq_len; ++c) {
                ptr[r * seq_len + c] = (c <= r);
            }
        }
    } else if (mask_type == ov::element::i64) {
        int64_t* ptr = mask.data<int64_t>();
        for (size_t r = 0; r < seq_len; ++r) {
            for (size_t c = 0; c < seq_len; ++c) {
                ptr[r * seq_len + c] = (c <= r) ? 1 : 0;
            }
        }
    } else if (mask_type == ov::element::i32) {
        int32_t* ptr = mask.data<int32_t>();
        for (size_t r = 0; r < seq_len; ++r) {
            for (size_t c = 0; c < seq_len; ++c) {
                ptr[r * seq_len + c] = (c <= r) ? 1 : 0;
            }
        }
    } else {
        OPENVINO_THROW("Unsupported attention_mask element type for DFlash token-major prefill");
    }

    return mask;
}

ov::Tensor to_batch_major_3d(const ov::Tensor& tensor) {
    const auto shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Expected rank-3 tensor for layout normalization");
    if (shape[0] == 1) {
        return tensor;
    }
    OPENVINO_ASSERT(shape[1] == 1, "Expected shape [N,1,C] or [1,N,C] for layout normalization");

    const size_t n = shape[0];
    const size_t c = shape[2];
    ov::Tensor normalized(tensor.get_element_type(), {1, n, c});
    const size_t bytes_per_row = c * tensor.get_element_type().size();
    for (size_t i = 0; i < n; ++i) {
        const char* src = static_cast<const char*>(tensor.data()) + i * bytes_per_row;
        char* dst = static_cast<char*>(normalized.data()) + i * bytes_per_row;
        std::memcpy(dst, src, bytes_per_row);
    }
    return normalized;
}

bool has_sdpa_nodes(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return false;
    }

    for (const auto& node : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node) != nullptr) {
            return true;
        }
    }
    return false;
}

bool is_token_id_row_tensor(const ov::Tensor& tensor) {
    const auto shape = tensor.get_shape();
    return tensor.get_element_type() == ov::element::i64 && shape.size() == 2 && shape[0] == 1;
}

bool is_embedding_row_tensor(const ov::Tensor& tensor) {
    const auto shape = tensor.get_shape();
    return (tensor.get_element_type() == ov::element::f16 || tensor.get_element_type() == ov::element::f32) && shape.size() == 3 &&
           shape[0] == 1;
}

bool is_truthy_env_flag(const char* value) {
    if (value == nullptr) {
        return false;
    }

    const std::string flag = value;
    return !(flag == "0" || flag == "false" || flag == "FALSE" || flag == "off" || flag == "OFF");
}

bool dflash_direct_trace_enabled() {
    static const bool enabled = []() {
        return is_truthy_env_flag(std::getenv("OPENVINO_GENAI_DFLASH_TRACE"));
    }();

    return enabled;
}

constexpr size_t kInvalidOutputIndex = std::numeric_limits<size_t>::max();

}  // namespace

namespace ov::genai {

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                   const ov::genai::ModelDesc& draft_model_desc,
                                                                   const std::vector<int32_t>& hidden_layers_to_abstract) {
    const auto main_scheduler_config = main_model_desc.scheduler_config;
    const auto draft_scheduler_config =
        (draft_model_desc.scheduler_config == SchedulerConfig()) ? main_scheduler_config : draft_model_desc.scheduler_config;
    m_tokenizer = main_model_desc.tokenizer;
    m_main_model = main_model_desc.model;
    m_main_direct_model = m_main_model;
    m_draft_model = draft_model_desc.model;
    m_main_device = main_model_desc.device;
    m_main_properties = main_model_desc.properties;
    m_main_scheduler_config = main_scheduler_config;
    m_main_generation_config = main_model_desc.generation_config;
    m_hidden_layers_to_abstract = hidden_layers_to_abstract;

    auto main_model = m_main_model;
    OPENVINO_ASSERT(main_model != nullptr, "Main model cannot be null for DFlashDecodingImpl");

    if (main_model_desc.inputs_embedder) {
        m_main_direct_model = m_main_direct_model->clone();
        utils::dflash::transform_hidden_state(m_main_direct_model, m_hidden_layers_to_abstract);
        remove_deepstack_transpose(m_main_direct_model);
    }

    utils::dflash::transform_hidden_state(main_model, m_hidden_layers_to_abstract);
    remove_deepstack_transpose(main_model);

    if (has_sdpa_nodes(main_model)) {
        bool allow_score_aggregation = true;
        bool allow_xattention = false;
        ov::pass::SDPAToPagedAttention(main_scheduler_config.use_cache_eviction,
                                       main_scheduler_config.use_cache_eviction,
                                       allow_score_aggregation,
                                       allow_xattention)
            .run_on_model(main_model);
        utils::apply_gather_before_matmul_transformation(main_model);
    }

    ov::AnyMap draft_properties = draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;
    draft_properties.erase("dflash_mode");
    draft_properties.erase("skip_sdpa_to_paged_attention");
    draft_properties.erase("hidden_layers_list");

    std::string draft_device = draft_model_desc.device.empty() ? m_main_device : draft_model_desc.device;

    if (main_model_desc.inputs_embedder) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_vision_registry = std::make_shared<VisionRegistry>();
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(main_model,
                                                                                          m_inputs_embedder,
                                                                                          main_model_desc.tokenizer,
                                                                                          main_model_desc.generation_config,
                                                                                          main_scheduler_config,
                                                                                          m_main_device,
                                                                                          m_main_properties,
                                                                                          true);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(main_model,
                                                                                          main_model_desc.tokenizer,
                                                                                          main_model_desc.generation_config,
                                                                                          main_scheduler_config,
                                                                                          m_main_device,
                                                                                          m_main_properties,
                                                                                          true);
    }

    if (m_draft_model && has_port_named(m_draft_model->inputs(), "input_ids")) {
        m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(m_draft_model,
                                                                                           draft_model_desc.tokenizer,
                                                                                           draft_model_desc.generation_config,
                                                                                           draft_scheduler_config,
                                                                                           draft_device,
                                                                                           draft_properties,
                                                                                           false);
        m_draft_speculative_pipeline_ready = true;
    }

    if (m_main_direct_model && m_draft_model) {
        try {
            auto main_compiled = utils::singleton_core().compile_model(m_main_direct_model, m_main_device, m_main_properties);
            auto draft_compiled = utils::singleton_core().compile_model(m_draft_model, draft_device, draft_properties);
            m_main_request = main_compiled.create_infer_request();
            m_draft_request = draft_compiled.create_infer_request();
            m_main_kv_axes_pos = utils::get_kv_axes_pos(m_main_direct_model);

            const auto main_outputs = m_main_direct_model->outputs();
            m_main_logits_output_index = 0;
            if (auto logits_port = find_port_by_name(main_outputs, "logits"); logits_port.has_value()) {
                m_main_logits_output_name = first_tensor_name_or_empty(logits_port.value());
                if (auto logits_idx = find_port_index_by_name(main_outputs, "logits"); logits_idx.has_value()) {
                    m_main_logits_output_index = logits_idx.value();
                }
            } else if (!main_outputs.empty()) {
                m_main_logits_output_name = first_tensor_name_or_empty(main_outputs[0]);
            }

            m_main_hidden_output_index = kInvalidOutputIndex;
            if (auto hidden_port = find_port_by_name(main_outputs, "hidden_states"); hidden_port.has_value()) {
                m_main_hidden_output_name = first_tensor_name_or_empty(hidden_port.value());
                if (auto hidden_idx = find_port_index_by_name(main_outputs, "hidden_states"); hidden_idx.has_value()) {
                    m_main_hidden_output_index = hidden_idx.value();
                }
            } else if (auto last_hidden_port = find_port_by_name(main_outputs, "last_hidden_state"); last_hidden_port.has_value()) {
                m_main_hidden_output_name = first_tensor_name_or_empty(last_hidden_port.value());
                if (auto hidden_idx = find_port_index_by_name(main_outputs, "last_hidden_state"); hidden_idx.has_value()) {
                    m_main_hidden_output_index = hidden_idx.value();
                }
            } else if (main_outputs.size() > 1) {
                m_main_hidden_output_index = (m_main_logits_output_index == 0) ? 1 : 0;
            }

            const auto draft_outputs = m_draft_model->outputs();
            m_draft_logits_output_index = 0;
            if (auto logits_port = find_port_by_name(draft_outputs, "logits"); logits_port.has_value()) {
                m_draft_logits_output_name = first_tensor_name_or_empty(logits_port.value());
                if (auto logits_idx = find_port_index_by_name(draft_outputs, "logits"); logits_idx.has_value()) {
                    m_draft_logits_output_index = logits_idx.value();
                }
            } else if (!draft_outputs.empty()) {
                m_draft_logits_output_name = first_tensor_name_or_empty(draft_outputs[0]);
            }

            const bool draft_has_required_inputs =
                has_port_named(m_draft_model->inputs(), "position_ids") && has_port_named(m_draft_model->inputs(), "attention_mask") &&
                has_port_named(m_draft_model->inputs(), "noise_embedding") && has_port_named(m_draft_model->inputs(), "target_hidden");

            const bool main_has_decode_inputs =
                (has_port_named(m_main_direct_model->inputs(), "input_ids") ||
                 has_port_named(m_main_direct_model->inputs(), "inputs_embeds")) &&
                has_port_named(m_main_direct_model->inputs(), "position_ids");
            const bool main_has_prefill_inputs = has_port_named(m_main_direct_model->inputs(), "input_ids") ||
                                                 has_port_named(m_main_direct_model->inputs(), "inputs_embeds");

            const bool main_has_logits_output = !main_outputs.empty();
            const bool main_has_hidden_output = m_main_hidden_output_index != kInvalidOutputIndex;
            const bool draft_has_logits_output = !draft_outputs.empty();
            m_direct_dflash_ready = draft_has_required_inputs && main_has_decode_inputs && main_has_prefill_inputs &&
                                    main_has_logits_output && main_has_hidden_output &&
                                    draft_has_logits_output;

            std::ostringstream main_inputs_desc;
            for (size_t i = 0; i < m_main_direct_model->inputs().size(); ++i) {
                if (i > 0) {
                    main_inputs_desc << " | ";
                }
                main_inputs_desc << describe_port(m_main_direct_model->inputs()[i]);
            }
            GENAI_INFO("DFLASH_MAIN_INPUTS %s", main_inputs_desc.str().c_str());

            GENAI_INFO("DFLASH_DIRECT_READY ready=%d draft_inputs=%d main_decode_inputs=%d main_prefill_inputs=%d "
                       "main_logits=%d main_hidden=%d draft_logits=%d main_logits_name=%s main_hidden_name=%s draft_logits_name=%s",
                       static_cast<int>(m_direct_dflash_ready),
                       static_cast<int>(draft_has_required_inputs),
                       static_cast<int>(main_has_decode_inputs),
                       static_cast<int>(main_has_prefill_inputs),
                       static_cast<int>(main_has_logits_output),
                       static_cast<int>(main_has_hidden_output),
                       static_cast<int>(draft_has_logits_output),
                       m_main_logits_output_name.c_str(),
                       m_main_hidden_output_name.c_str(),
                       m_draft_logits_output_name.c_str());
        } catch (const std::exception& ex) {
            GENAI_WARN("DFlash direct block decode init failed, fallback to CB path: " + std::string(ex.what()));
            m_direct_dflash_ready = false;
        }
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
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);

    ov::genai::GenerationConfig draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};

    std::optional<ov::Tensor> draft_input_ids = std::nullopt;
    if (is_token_id_row_tensor(input_ids)) {
        draft_input_ids = input_ids;
    } else if (is_embedding_row_tensor(input_ids) && prompt_ids.has_value() && is_token_id_row_tensor(prompt_ids.value())) {
        // For multimodal main requests, draft pipeline still consumes token ids.
        draft_input_ids = prompt_ids.value();
    }

    if (m_draft_speculative_pipeline_ready && m_draft_pipeline && draft_input_ids.has_value()) {
        m_draft_generations.insert({request_id,
                                    m_draft_pipeline->add_request(request_id,
                                                                  draft_input_ids.value(),
                                                                  draft_sampling_params,
                                                                  std::nullopt,
                                                                  std::nullopt)});
    }

    m_dflash_pending_requests[request_id] = DFlashPendingRequest{
        input_ids,
        draft_sampling_params,
        token_type_ids,
        prompt_ids,
    };

    return m_main_pipeline->add_request(
        request_id, input_ids, sampling_params, token_type_ids, prompt_ids, lm_extra_inputs);
}

GenerationHandle
ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(uint64_t request_id,
                                                            const std::string& prompt,
                                                            const ov::genai::GenerationConfig& sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);

    ov::genai::GenerationConfig draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};

    ov::Tensor input_ids = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false)).input_ids;
    if (m_draft_speculative_pipeline_ready && m_draft_pipeline) {
        m_draft_generations.insert(
            {request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params)});
    }
    m_dflash_pending_requests[request_id] = DFlashPendingRequest{input_ids, draft_sampling_params, std::nullopt, std::nullopt};

    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    bool has_draft_requests = false;
    {
        std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
        has_draft_requests = m_draft_speculative_pipeline_ready && !m_draft_generations.empty();
    }

    if (has_draft_requests && m_draft_pipeline) {
        // Reuse the validated speculative loop when the draft model can be scheduled via speculative pipeline.
        SpeculativeDecodingImpl::step();
    } else {
        // Main-only fallback: explicitly pull awaiting requests because speculative pipeline overrides _pull_awaiting_requests.
        m_main_pipeline->pull_awaiting_requests();
        m_main_pipeline->step();
    }

    if (!m_main_pipeline->has_non_finished_requests()) {
        std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
        m_dflash_pending_requests.clear();
        m_draft_generations.clear();
    }
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
    if (!m_direct_dflash_ready) {
        OPENVINO_THROW("DFlash direct block decode is not ready; check main/draft model outputs and required ports.");
    }
    if (!std::holds_alternative<std::monostate>(streamer)) {
        OPENVINO_THROW("DFlash direct block decode does not support streaming callbacks.");
    }
    if (!sampling_params.empty() && !sampling_params.front().is_greedy_decoding()) {
        OPENVINO_THROW("DFlash direct block decode requires greedy decoding.");
    }
    if (token_type_ids.has_value() && !token_type_ids->empty()) {
        OPENVINO_THROW("DFlash direct block decode does not support token_type_ids.");
    }
    OPENVINO_ASSERT(input_ids.size() == 1 && sampling_params.size() == 1,
                    "DFlash direct block decode supports batch size=1");

    const auto& input = input_ids.front();
    const bool is_token_id_input =
        input.get_element_type() == ov::element::i64 && input.get_shape().size() == 2 && input.get_shape()[0] == 1;
    const bool is_embedding_input =
        (input.get_element_type() == ov::element::f16 || input.get_element_type() == ov::element::f32) &&
        input.get_shape().size() == 3 && input.get_shape()[0] == 1;
    if (!is_token_id_input && !is_embedding_input) {
        OPENVINO_THROW("DFlash direct block decode expects input_ids or inputs_embeds with batch size 1.");
    }

    std::optional<ov::Tensor> token_type_id = std::nullopt;
    if (token_type_ids.has_value() && !token_type_ids->empty()) {
        token_type_id = token_type_ids->front();
    }
    std::optional<std::pair<ov::Tensor, std::optional<int64_t>>> position_id = std::nullopt;
    if (position_ids.has_value() && !position_ids->empty()) {
        position_id = position_ids->front();
    }
    std::optional<ov::Tensor> prompt_id = std::nullopt;
    if (prompt_ids.has_value() && !prompt_ids->empty()) {
        prompt_id = prompt_ids->front();
    }
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt;
    if (lm_extra_inputs_list.has_value() && !lm_extra_inputs_list->empty()) {
        lm_extra_inputs = lm_extra_inputs_list->front();
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(1);
    results.emplace_back(direct_block_decode_generate(input_ids.front(),
                                                      sampling_params.front(),
                                                      streamer,
                                                      token_type_id,
                                                      position_id,
                                                      prompt_id,
                                                      lm_extra_inputs));
    return results;
}

EncodedGenerationResult ContinuousBatchingPipeline::DFlashDecodingImpl::direct_block_decode_generate(
    const ov::Tensor& input_ids,
    const GenerationConfig& sampling_params,
    const StreamerVariant&,
    const std::optional<ov::Tensor>&,
    const std::optional<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids,
    const std::optional<ov::Tensor>&,
    const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs) {
    OPENVINO_ASSERT(m_inputs_embedder, "InputsEmbedder is required for DFlash direct block decode");

    auto embedding_model = m_inputs_embedder->get_embedding_model();
    OPENVINO_ASSERT(embedding_model, "Embedding model is required for DFlash direct block decode");

    GENAI_INFO("DFLASH_DIRECT_ENTER mode=inputs_embeds");

    m_main_request.reset_state();
    m_draft_request.reset_state();

    const auto input_shape = input_ids.get_shape();
    const bool is_embedding_prefill =
        input_shape.size() == 3 && (input_ids.get_element_type() == ov::element::f16 || input_ids.get_element_type() == ov::element::f32);
    const bool trace_direct_path = is_embedding_prefill && dflash_direct_trace_enabled();

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    auto& direct_raw_perf = m_perf_metrics.raw_metrics;
    auto& direct_main_raw_perf = m_perf_metrics.main_model_metrics.raw_metrics;
    auto& direct_draft_raw_perf = m_perf_metrics.draft_model_metrics.raw_metrics;

    const size_t prompt_len = input_shape[1];
    const size_t max_new_tokens = sampling_params.get_max_new_tokens(prompt_len);
    OPENVINO_ASSERT(max_new_tokens > 0, "max_new_tokens must be > 0 for DFlash decoding");

    size_t decode_start_pos = prompt_len;
    if (position_ids.has_value()) {
        const ov::Tensor& position_tensor = position_ids->first;
        const auto pos_shape = position_tensor.get_shape();
        int64_t next_pos = -1;
        if (pos_shape.size() == 2) {
            OPENVINO_ASSERT(pos_shape[0] == 1, "position_ids rank-2 tensor must have shape [1, seq] for DFlash direct mode");
            OPENVINO_ASSERT(pos_shape[1] == prompt_len, "position_ids length must match prompt length in DFlash direct mode");
            const int64_t* pos_ptr = position_tensor.data<const int64_t>();
            next_pos = pos_ptr[prompt_len - 1] + 1;
        } else if (pos_shape.size() == 3) {
            OPENVINO_ASSERT((pos_shape[1] == 1 && pos_shape[2] == prompt_len) ||
                                (pos_shape[2] == 1 && pos_shape[1] == prompt_len),
                            "position_ids rank-3 tensor must have shape [rows,1,seq] or [rows,seq,1] for DFlash direct mode");
            const int64_t* pos_ptr = position_tensor.data<const int64_t>();

            int64_t max_last_pos = std::numeric_limits<int64_t>::min();
            const size_t rows = pos_shape[0];
            if (pos_shape[1] == 1) {
                for (size_t r = 0; r < rows; ++r) {
                    const int64_t last_pos = pos_ptr[r * prompt_len + (prompt_len - 1)];
                    max_last_pos = std::max(max_last_pos, last_pos);
                }
            } else {
                for (size_t r = 0; r < rows; ++r) {
                    const int64_t last_pos = pos_ptr[r * prompt_len + (prompt_len - 1)];
                    max_last_pos = std::max(max_last_pos, last_pos);
                }
            }
            next_pos = max_last_pos + 1;
        } else {
            OPENVINO_THROW("position_ids rank ", pos_shape.size(), " is not supported in DFlash direct mode");
        }

        OPENVINO_ASSERT(next_pos > 0, "position_ids must be non-negative for DFlash direct mode");
        decode_start_pos = static_cast<size_t>(next_pos);
    }

    size_t block_size = sampling_params.num_assistant_tokens;
    if (block_size == 0) {
        block_size = 16;
    }
    if (block_size != 16) {
        GENAI_WARN("DFLASH block_size must be 16 for direct mode; overriding requested=%zu", block_size);
        block_size = 16;
    }

    const size_t max_length = prompt_len + max_new_tokens;
    std::vector<int64_t> output_ids(max_length + block_size, sampling_params.eos_token_id >= 0 ? sampling_params.eos_token_id : -1);
    if (!is_embedding_prefill) {
        const int64_t* prompt_ptr = input_ids.data<const int64_t>();
        std::copy(prompt_ptr, prompt_ptr + prompt_len, output_ids.begin());
    }

    std::optional<ov::Tensor> prefill_hidden;
    ov::Tensor prefill_logits;
    if (is_embedding_prefill) {
        prefill_logits = run_main_model_prefill_with_embeddings(input_ids, position_ids, lm_extra_inputs, prefill_hidden);
    } else {
        prefill_logits = run_main_model_step(input_ids, 0, lm_extra_inputs, prefill_hidden, true);
    }
    OPENVINO_ASSERT(prefill_hidden.has_value() && prefill_hidden->get_size() > 0,
                    "DFlash direct mode requires hidden-state output from main model");

    int64_t next_token = greedy_pick_last_token(prefill_logits, prefill_logits.get_shape()[1] - 1);
    output_ids[prompt_len] = next_token;
    ov::Tensor target_hidden = prefill_hidden.value();

    size_t direct_iteration_idx = 0;
    size_t total_draft_candidates_proposed = 0;
    size_t total_draft_candidates_accepted = 0;
    size_t total_draft_candidates_rejected = 0;

    auto is_stop_token = [&sampling_params, this](int64_t token) {
        if (sampling_params.eos_token_id >= 0 && token == sampling_params.eos_token_id) {
            return true;
        }
        if (sampling_params.stop_token_ids.find(token) != sampling_params.stop_token_ids.end()) {
            return true;
        }
        const int64_t tokenizer_eos = static_cast<int64_t>(m_tokenizer.get_eos_token_id());
        return tokenizer_eos >= 0 && token == tokenizer_eos;
    };

    size_t start = prompt_len;
    bool stopped = is_stop_token(next_token);
    while (!stopped && start < max_length) {
        const auto iteration_start = std::chrono::steady_clock::now();
        ++direct_iteration_idx;
        const size_t current_block_size = std::min(block_size, max_length - start);
        ov::Tensor block_output_ids(ov::element::i64, {1, current_block_size});
        int64_t* block_ptr = block_output_ids.data<int64_t>();

        block_ptr[0] = output_ids[start];
        for (size_t i = 1; i < current_block_size; ++i) {
            block_ptr[i] = sampling_params.eos_token_id >= 0 ? sampling_params.eos_token_id : 0;
        }

        std::optional<ov::Tensor> block_input_embeds = std::nullopt;
        {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(embedding_model->get_request_queue().get());
            EmbeddingsRequest& req = embeddings_request_guard.get();
            ov::Tensor noise_embedding = embedding_model->infer(req, block_output_ids);
            block_input_embeds = noise_embedding;

            ov::Tensor draft_logits = run_draft_model_step(noise_embedding, target_hidden, current_block_size);
            if (current_block_size > 1) {
                const size_t draft_seq_len = draft_logits.get_shape()[1];
                OPENVINO_ASSERT(draft_seq_len >= current_block_size - 1,
                                "Draft logits sequence length is smaller than required candidate window");
                for (size_t i = 1; i < current_block_size; ++i) {
                    const size_t pos = draft_seq_len - (current_block_size - i);
                    block_ptr[i] = greedy_pick_last_token(draft_logits, pos);
                }
            }
        }

        std::optional<ov::Tensor> block_hidden;
        std::vector<int64_t> posterior(current_block_size);
        for (size_t i = 0; i < current_block_size; ++i) {
            ov::Tensor step_input_ids(ov::element::i64, {1, 1});
            step_input_ids.data<int64_t>()[0] = block_ptr[i];

            std::optional<ov::Tensor> step_input_embeds = std::nullopt;
            if (is_embedding_prefill && block_input_embeds.has_value()) {
                const auto& full_embeds = block_input_embeds.value();
                const auto full_shape = full_embeds.get_shape();
                OPENVINO_ASSERT(full_shape.size() == 3 && full_shape[1] >= (i + 1),
                                "Invalid block input embeds shape for DFlash direct decode");
                auto [emb_start_coord, emb_end_coord] = utils::make_roi(full_shape, 1, i, i + 1);
                step_input_embeds = ov::Tensor(full_embeds, emb_start_coord, emb_end_coord);
            }

            std::optional<ov::Tensor> step_hidden;
            ov::Tensor step_logits = run_main_model_step(step_input_ids,
                                                         decode_start_pos + (start - prompt_len) + i,
                                                         lm_extra_inputs,
                                                         step_hidden,
                                                         true,
                                                         step_input_embeds);
            OPENVINO_ASSERT(step_hidden.has_value() && step_hidden->get_size() > 0,
                            "DFlash direct mode requires hidden-state output for decode stage");

            posterior[i] = greedy_pick_last_token(step_logits, step_logits.get_shape()[1] - 1);

            if (!block_hidden.has_value()) {
                const auto step_hidden_shape = step_hidden->get_shape();
                OPENVINO_ASSERT(step_hidden_shape.size() == 3 && step_hidden_shape[1] == 1,
                                "Expected step hidden state shape [1,1,H] in DFlash direct decode");
                block_hidden = ov::Tensor(step_hidden->get_element_type(), {1, current_block_size, step_hidden_shape[2]});
            }

            const auto step_hidden_shape = step_hidden->get_shape();
            const size_t bytes_per_step_hidden = step_hidden_shape[2] * step_hidden->get_element_type().size();
            char* dst = static_cast<char*>(block_hidden->data()) + i * bytes_per_step_hidden;
            const char* src = static_cast<const char*>(step_hidden->data());
            std::memcpy(dst, src, bytes_per_step_hidden);
        }

        OPENVINO_ASSERT(block_hidden.has_value() && block_hidden->get_size() > 0,
                        "DFlash direct mode requires hidden-state output for decode stage");

        size_t acceptance_len = 0;
        for (size_t i = 1; i < current_block_size; ++i) {
            if (block_ptr[i] != posterior[i - 1]) {
                break;
            }
            ++acceptance_len;
        }

        const size_t draft_candidates_proposed = current_block_size > 0 ? (current_block_size - 1) : 0;
        const size_t draft_candidates_accepted = acceptance_len;
        const size_t draft_candidates_rejected = draft_candidates_proposed - draft_candidates_accepted;
        total_draft_candidates_proposed += draft_candidates_proposed;
        total_draft_candidates_accepted += draft_candidates_accepted;
        total_draft_candidates_rejected += draft_candidates_rejected;

        const size_t accepted_len = acceptance_len + 1;
        for (size_t i = 0; i < accepted_len; ++i) {
            output_ids[start + i] = block_ptr[i];
        }

        if (start + accepted_len < max_length) {
            output_ids[start + accepted_len] = posterior[acceptance_len];
            stopped = is_stop_token(output_ids[start + accepted_len]);
        }

        const size_t tokens_to_trim = current_block_size - accepted_len;
        trim_main_kv_cache(tokens_to_trim, start + current_block_size);
        target_hidden = slice_hidden_prefix(block_hidden.value(), accepted_len);

        if (direct_iteration_idx < 5) {
            GENAI_INFO("DFLASH_DIRECT_ITER mode=inputs_embeds iter=%zu block=%zu proposed=%zu acceptance_len=%zu accepted_len=%zu draft_accepted=%zu rejected=%zu trim=%zu, start=%zu", 
                   direct_iteration_idx,
                   current_block_size,
                   draft_candidates_proposed,
                   acceptance_len,
                   accepted_len,
                   draft_candidates_accepted,
                   draft_candidates_rejected,
                   tokens_to_trim,
                   start);
        }

        const auto iteration_end = std::chrono::steady_clock::now();
        const auto iteration_duration_us =
            MicroSeconds(PerfMetrics::get_microsec(iteration_end - iteration_start));
        direct_raw_perf.m_durations.emplace_back(iteration_duration_us);
        direct_raw_perf.m_token_infer_durations.emplace_back(iteration_duration_us);
        direct_raw_perf.m_new_token_times.emplace_back(iteration_end);
        direct_raw_perf.m_batch_sizes.emplace_back(accepted_len);
        direct_raw_perf.m_inference_durations[0] += iteration_duration_us;

        direct_main_raw_perf.m_durations.emplace_back(iteration_duration_us);
        direct_main_raw_perf.m_batch_sizes.emplace_back(draft_candidates_accepted);
        direct_main_raw_perf.m_new_token_times.emplace_back(iteration_end);
        direct_main_raw_perf.m_inference_durations[0] += iteration_duration_us;

        direct_draft_raw_perf.m_batch_sizes.emplace_back(draft_candidates_proposed);

        for (size_t i = 0; i < accepted_len; ++i) {
            if (is_stop_token(output_ids[start + i])) {
                stopped = true;
                break;
            }
        }

        start += accepted_len;
    }

    size_t end_pos = std::min(start, max_length);
    for (size_t i = prompt_len; i < end_pos; ++i) {
        if (is_stop_token(output_ids[i])) {
            end_pos = i + 1;
            break;
        }
    }

    std::vector<int64_t> generated_ids;
    if (sampling_params.echo) {
        generated_ids.assign(output_ids.begin(), output_ids.begin() + end_pos);
    } else {
        generated_ids.assign(output_ids.begin() + prompt_len, output_ids.begin() + end_pos);
    }

    EncodedGenerationResult result;
    result.m_request_id = 0;
    result.m_generation_ids = {std::move(generated_ids)};
    result.m_scores = {0.0f};
    result.m_status = GenerationStatus::FINISHED;
    result.perf_metrics = m_perf_metrics;

    if (trace_direct_path) {
        const double acceptance_pct =
            total_draft_candidates_proposed > 0
                ? (100.0 * static_cast<double>(total_draft_candidates_accepted) /
                   static_cast<double>(total_draft_candidates_proposed))
                : 0.0;
        GENAI_INFO("DFLASH_DIRECT_SUMMARY mode=inputs_embeds iters=%zu proposed=%zu accepted=%zu rejected=%zu acceptance=%.2f%%", 
                   direct_iteration_idx,
                   total_draft_candidates_proposed,
                   total_draft_candidates_accepted,
                   total_draft_candidates_rejected,
                   acceptance_pct);
    }

    result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_perf_metrics);
    return result;
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::run_main_model_step(
    const ov::Tensor& token_ids,
    size_t start_pos,
    const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs,
    std::optional<ov::Tensor>& hidden_state,
    bool apply_extra_inputs,
    const std::optional<ov::Tensor>& input_embeds) {
    const size_t seq_len = token_ids.get_shape()[1];

    const bool has_input_ids = has_port_named(m_main_direct_model->inputs(), "input_ids");
    const bool has_inputs_embeds = has_port_named(m_main_direct_model->inputs(), "inputs_embeds");

    if (input_embeds.has_value() && has_inputs_embeds) {
        OPENVINO_ASSERT(input_embeds.has_value(), "inputs_embeds tensor is required for DFlash direct decode step");
        m_main_request.set_tensor("inputs_embeds", input_embeds.value());
    } else if (has_input_ids) {
        m_main_request.set_tensor("input_ids", token_ids);
    } else if (has_inputs_embeds) {
        OPENVINO_ASSERT(input_embeds.has_value(), "inputs_embeds tensor is required for DFlash direct decode step");
        m_main_request.set_tensor("inputs_embeds", input_embeds.value());
    } else {
        OPENVINO_THROW("DFlash direct mode expected main model to have input_ids or inputs_embeds");
    }

    if (auto pos_input = find_port_by_name(m_main_direct_model->inputs(), "position_ids"); pos_input.has_value()) {
        m_main_request.set_tensor("position_ids", build_position_ids_like(pos_input.value(), start_pos, seq_len));
    }

    if (auto mask_input = find_port_by_name(m_main_direct_model->inputs(), "attention_mask"); mask_input.has_value()) {
        m_main_request.set_tensor("attention_mask", build_attention_mask_like(mask_input.value(), start_pos + seq_len));
    }

    if (has_port_named(m_main_direct_model->inputs(), "beam_idx")) {
        ov::Tensor beam_idx(ov::element::i32, {1});
        beam_idx.data<int32_t>()[0] = 0;
        m_main_request.set_tensor("beam_idx", beam_idx);
    }

    if (apply_extra_inputs && lm_extra_inputs.has_value()) {
        if (input_embeds.has_value()) {
            const auto embed_shape = input_embeds.value().get_shape();
            OPENVINO_ASSERT(embed_shape.size() == 3, "inputs_embeds for direct decode must have rank 3 [B,S,H]");
            const size_t decode_seq_len = embed_shape[1];

            const bool has_visual_pos_mask = has_port_named(m_main_direct_model->inputs(), "visual_pos_mask");
            const bool has_visual_pos_masks = has_port_named(m_main_direct_model->inputs(), "visual_pos_masks");
            if (has_visual_pos_mask || has_visual_pos_masks) {
                ov::Tensor decode_visual_pos_mask(ov::element::boolean, {1, decode_seq_len});
                std::fill_n(decode_visual_pos_mask.data<bool>(), decode_visual_pos_mask.get_size(), false);
                if (has_visual_pos_mask) {
                    m_main_request.set_tensor("visual_pos_mask", decode_visual_pos_mask);
                }
                if (has_visual_pos_masks) {
                    m_main_request.set_tensor("visual_pos_masks", decode_visual_pos_mask);
                }
            }

            if (has_port_named(m_main_direct_model->inputs(), "deepstack_visual_embeds")) {
                size_t deepstack_layers = 1;
                size_t hidden_size = embed_shape[2];
                ov::element::Type deepstack_type = input_embeds.value().get_element_type();

                auto deepstack_it = lm_extra_inputs->find("deepstack_visual_embeds");
                if (deepstack_it != lm_extra_inputs->end() && deepstack_it->second) {
                    const ov::Tensor& deepstack_ref = deepstack_it->second;
                    const auto deepstack_shape = deepstack_ref.get_shape();
                    if (deepstack_shape.size() == 3) {
                        deepstack_layers = deepstack_shape[0];
                        hidden_size = deepstack_shape[2];
                        deepstack_type = deepstack_ref.get_element_type();
                    }
                }

                ov::Tensor decode_deepstack_embeds(deepstack_type, {deepstack_layers, decode_seq_len, hidden_size});
                std::memset(decode_deepstack_embeds.data(), 0, decode_deepstack_embeds.get_byte_size());
                m_main_request.set_tensor("deepstack_visual_embeds", decode_deepstack_embeds);
            }
        }

        for (const auto& [name, tensor] : lm_extra_inputs.value()) {
            if (input_embeds.has_value() && (name == "deepstack_visual_embeds" || name == "visual_pos_masks" || name == "visual_pos_mask")) {
                continue;
            }
            if (tensor && has_port_named(m_main_direct_model->inputs(), name)) {
                m_main_request.set_tensor(name, tensor);
            }
        }
    }

    if (dflash_direct_trace_enabled()) {
        std::ostringstream decode_inputs_desc;
        decode_inputs_desc << "DFLASH_MAIN_DECODE_INPUTS start_pos=" << start_pos << " seq_len=" << seq_len;

        auto append_shape_if_present = [&](const std::string& name) {
            if (has_port_named(m_main_direct_model->inputs(), name)) {
                const ov::Shape shape = m_main_request.get_tensor(name).get_shape();
                decode_inputs_desc << " " << name << "=" << shape_to_string(shape);
            }
        };

        append_shape_if_present("input_ids");
        append_shape_if_present("inputs_embeds");
        append_shape_if_present("position_ids");
        append_shape_if_present("attention_mask");
        append_shape_if_present("beam_idx");
        append_shape_if_present("visual_pos_mask");
        append_shape_if_present("visual_pos_masks");
        append_shape_if_present("deepstack_visual_embeds");

        GENAI_INFO("%s", decode_inputs_desc.str().c_str());
    }

    m_main_request.infer();

    hidden_state = std::nullopt;
    if (!m_main_hidden_output_name.empty()) {
        hidden_state = m_main_request.get_tensor(m_main_hidden_output_name);
    } else if (m_main_hidden_output_index != kInvalidOutputIndex) {
        hidden_state = m_main_request.get_output_tensor(m_main_hidden_output_index);
    }

    if (hidden_state.has_value()) {
        hidden_state = to_batch_major_3d(hidden_state.value());
    }

    ov::Tensor logits;
    if (!m_main_logits_output_name.empty()) {
        logits = m_main_request.get_tensor(m_main_logits_output_name);
    } else {
        logits = m_main_request.get_output_tensor(m_main_logits_output_index);
    }
    return to_batch_major_3d(logits);
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::run_main_model_prefill_with_embeddings(
    const ov::Tensor& inputs_embeds,
    const std::optional<std::pair<ov::Tensor, std::optional<int64_t>>>& position_ids,
    const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs,
    std::optional<ov::Tensor>& hidden_state) {
    const size_t seq_len = inputs_embeds.get_shape()[1];
    const ov::Tensor prefill_inputs_embeds = inputs_embeds;

    if (auto embeds_input = find_port_by_name(m_main_direct_model->inputs(), "inputs_embeds"); embeds_input.has_value()) {
        m_main_request.set_tensor("inputs_embeds", prefill_inputs_embeds);
        GENAI_INFO("DFLASH_PREFILL_INPUT name=inputs_embeds shape=%s", shape_to_string(prefill_inputs_embeds.get_shape()).c_str());
    } else {
        OPENVINO_THROW("DFlash direct mode expected main model to have inputs_embeds for multimodal prefill");
    }

    if (auto pos_input = find_port_by_name(m_main_direct_model->inputs(), "position_ids"); pos_input.has_value()) {
        if (position_ids.has_value()) {
            ov::Tensor adapted_position_ids = adapt_position_ids_to_input(position_ids->first, pos_input.value());
            m_main_request.set_tensor("position_ids", adapted_position_ids);
            GENAI_INFO("DFLASH_PREFILL_INPUT name=position_ids shape=%s", shape_to_string(adapted_position_ids.get_shape()).c_str());
        } else {
            ov::Tensor built_position_ids = build_position_ids_like(pos_input.value(), 0, seq_len);
            m_main_request.set_tensor("position_ids", built_position_ids);
            GENAI_INFO("DFLASH_PREFILL_INPUT name=position_ids shape=%s", shape_to_string(built_position_ids.get_shape()).c_str());
        }
    }

    if (auto mask_input = find_port_by_name(m_main_direct_model->inputs(), "attention_mask"); mask_input.has_value()) {
        ov::Tensor attention_mask = build_attention_mask_like(mask_input.value(), seq_len);
        m_main_request.set_tensor("attention_mask", attention_mask);
        GENAI_INFO("DFLASH_PREFILL_INPUT name=attention_mask shape=%s", shape_to_string(attention_mask.get_shape()).c_str());
    }

    if (has_port_named(m_main_direct_model->inputs(), "beam_idx")) {
        const size_t beam_batch = prefill_inputs_embeds.get_shape()[0];
        ov::Tensor beam_idx(ov::element::i32, {beam_batch});
        std::fill_n(beam_idx.data<int32_t>(), beam_batch, 0);
        m_main_request.set_tensor("beam_idx", beam_idx);
    }

    if (lm_extra_inputs.has_value()) {
        for (const auto& [name, tensor] : lm_extra_inputs.value()) {
            if (!tensor) {
                continue;
            }

            if (name == "visual_pos_masks" || name == "visual_pos_mask") {
                ov::Tensor normalized_visual_mask = normalize_visual_pos_mask_tensor(tensor);
                if (has_port_named(m_main_direct_model->inputs(), "visual_pos_mask")) {
                    m_main_request.set_tensor("visual_pos_mask", normalized_visual_mask);
                    GENAI_INFO("DFLASH_PREFILL_INPUT name=visual_pos_mask shape=%s",
                               shape_to_string(normalized_visual_mask.get_shape()).c_str());
                    continue;
                }
                if (has_port_named(m_main_direct_model->inputs(), "visual_pos_masks")) {
                    m_main_request.set_tensor("visual_pos_masks", normalized_visual_mask);
                    GENAI_INFO("DFLASH_PREFILL_INPUT name=visual_pos_masks shape=%s",
                               shape_to_string(normalized_visual_mask.get_shape()).c_str());
                    continue;
                }
            }

            if (has_port_named(m_main_direct_model->inputs(), name)) {
                m_main_request.set_tensor(name, tensor);
                GENAI_INFO("DFLASH_PREFILL_INPUT name=%s shape=%s", name.c_str(), shape_to_string(tensor.get_shape()).c_str());
            }
        }
    }

    m_main_request.infer();

    hidden_state = std::nullopt;
    if (!m_main_hidden_output_name.empty()) {
        hidden_state = m_main_request.get_tensor(m_main_hidden_output_name);
    } else if (m_main_hidden_output_index != kInvalidOutputIndex) {
        hidden_state = m_main_request.get_output_tensor(m_main_hidden_output_index);
    }

    if (hidden_state.has_value()) {
        hidden_state = to_batch_major_3d(hidden_state.value());
    }

    ov::Tensor logits;
    if (!m_main_logits_output_name.empty()) {
        logits = m_main_request.get_tensor(m_main_logits_output_name);
    } else {
        logits = m_main_request.get_output_tensor(m_main_logits_output_index);
    }
    return to_batch_major_3d(logits);
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::run_draft_model_step(const ov::Tensor& noise_embedding,
                                                                                 const ov::Tensor& target_hidden,
                                                                                 size_t block_size) {
    const size_t seq_len = target_hidden.get_shape()[1] + noise_embedding.get_shape()[1];
    auto position_ids = build_i64_arange_row(0, seq_len);
    auto mask_input = find_port_by_name(m_draft_model->inputs(), "attention_mask");
    OPENVINO_ASSERT(mask_input.has_value(), "DFlash draft model must have attention_mask input");

    m_draft_request.set_tensor("position_ids", position_ids);
    m_draft_request.set_tensor("attention_mask", build_attention_mask_like(mask_input.value(), seq_len));
    m_draft_request.set_tensor("noise_embedding", noise_embedding);
    m_draft_request.set_tensor("target_hidden", target_hidden);
    m_draft_request.infer();

    ov::Tensor logits;
    if (!m_draft_logits_output_name.empty()) {
        logits = m_draft_request.get_tensor(m_draft_logits_output_name);
    } else {
        logits = m_draft_request.get_output_tensor(m_draft_logits_output_index);
    }
    if (block_size <= 1) {
        return logits;
    }

    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] >= block_size - 1,
                    "Invalid draft logits shape for DFlash block decode");

    auto [start_coord, end_coord] = utils::make_roi(shape, 1, shape[1] - (block_size - 1), shape[1]);
    return ov::Tensor(logits, start_coord, end_coord);
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::trim_main_kv_cache(size_t tokens_to_trim,
                                                                         size_t current_sequence_len) {
    if (tokens_to_trim == 0 || current_sequence_len == 0 || tokens_to_trim >= current_sequence_len) {
        return;
    }

    if (m_main_device == "NPU") {
        return;
    }

    utils::KVCacheState state;
    state.num_tokens_to_trim = tokens_to_trim;
    state.seq_length_axis = m_main_kv_axes_pos.seq_len;
    state.reset_mem_state = false;
    utils::trim_kv_cache(m_main_request, state, {});
}

int64_t ContinuousBatchingPipeline::DFlashDecodingImpl::greedy_pick_last_token(const ov::Tensor& logits,
                                                                                size_t position_idx) {
    const auto shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Expected logits shape [1, seq, vocab]");
    OPENVINO_ASSERT(position_idx < shape[1], "position_idx is out of range for logits tensor");

    const size_t vocab_size = shape[2];
    const float* data = logits.data<const float>();
    const float* row = data + position_idx * vocab_size;
    const auto max_it = std::max_element(row, row + vocab_size);
    return static_cast<int64_t>(std::distance(row, max_it));
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::build_i64_arange_row(size_t start, size_t length) {
    ov::Tensor tensor(ov::element::i64, {1, length});
    int64_t* ptr = tensor.data<int64_t>();
    for (size_t i = 0; i < length; ++i) {
        ptr[i] = static_cast<int64_t>(start + i);
    }
    return tensor;
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::build_position_ids_like(const ov::Output<ov::Node>& position_input,
                                                                                    size_t start,
                                                                                    size_t length) {
    const auto pshape = position_input.get_partial_shape();
    if (pshape.rank().is_static() && pshape.rank().get_length() == 3) {
        size_t rows = 3;
        if (pshape[0].is_static()) {
            rows = static_cast<size_t>(pshape[0].get_length());
        }
        ov::Tensor tensor(ov::element::i64, {rows, length, 1});
        int64_t* ptr = tensor.data<int64_t>();
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i < length; ++i) {
                ptr[r * length + i] = static_cast<int64_t>(start + i);
            }
        }
        return tensor;
    }

    return build_i64_arange_row(start, length);
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::build_attention_mask_like(const ov::Output<ov::Node>& mask_input,
                                                                                      size_t length) {
    const auto mask_type = mask_input.get_element_type();
    ov::Tensor mask(mask_type, {1, length});

    if (mask_type == ov::element::boolean) {
        std::fill_n(mask.data<bool>(), length, true);
    } else if (mask_type == ov::element::i64) {
        std::fill_n(mask.data<int64_t>(), length, 1);
    } else if (mask_type == ov::element::i32) {
        std::fill_n(mask.data<int32_t>(), length, 1);
    } else {
        OPENVINO_THROW("Unsupported attention_mask element type for DFlash direct path");
    }

    return mask;
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::slice_hidden_prefix(const ov::Tensor& hidden_state,
                                                                                size_t prefix_len) {
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && prefix_len > 0 && prefix_len <= shape[1],
                    "Invalid hidden state shape or prefix length for DFlash update");
    auto [start_coord, end_coord] = utils::make_roi(shape, 1, 0, prefix_len);
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

}  // namespace ov::genai
