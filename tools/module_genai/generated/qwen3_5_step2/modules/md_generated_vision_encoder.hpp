// Auto-generated skeleton
#pragma once

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

namespace ov {
namespace genai {
namespace module {

class GeneratedVisionEncoderModule : public IBaseModule {
    DeclareModuleConstructor(GeneratedVisionEncoderModule);

private:
    bool initialize();
    void run_with_python_reference();
};

REGISTER_MODULE_CONFIG(GeneratedVisionEncoderModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
