// Auto-generated parity template from existing module implementation
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "visual_language/embedding_model.hpp"


namespace ov {
namespace genai {
namespace module {

class GeneratedTextEmbeddingModule : public IBaseModule {
    DeclareModuleConstructor(GeneratedTextEmbeddingModule);

private:
    bool initialize();
    std::shared_ptr<ov::genai::EmbeddingsModel> m_embedding_model;
};

REGISTER_MODULE_CONFIG(GeneratedTextEmbeddingModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
