// Auto-generated skeleton
#include "md_generated_image_preprocessor.hpp"

#include "module_genai/module_factory.hpp"
#include <cstdlib>

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(GeneratedImagePreprocessorModule);

void GeneratedImagePreprocessorModule::print_static_config() {
    std::cout << R"(
  image_preprocessor:
    type: "GeneratedImagePreprocessorModule"
    device: "GPU"
    # TODO: declare inputs/outputs/params
    )" << std::endl;
}

GeneratedImagePreprocessorModule::GeneratedImagePreprocessorModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize GeneratedImagePreprocessorModule");
    }
}

GeneratedImagePreprocessorModule::~GeneratedImagePreprocessorModule() {}

bool GeneratedImagePreprocessorModule::initialize() {
    return true;
}

void GeneratedImagePreprocessorModule::run_with_python_reference() {
    // GPU阶段建议：通过 Python 原始实现做对齐。
    // TODO: 使用 pybind11 或 subprocess + 文件交换实现 I/O 映射。
    // 示例：std::system("python3 qwen3_5_reference_bridge.py --stage image_preprocessor");
}

void GeneratedImagePreprocessorModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    if (module_desc->device == "GPU") {
        run_with_python_reference();
        return;
    }

    // TODO: 在此填充 CPU 参考实现。
}

}  // namespace module
}  // namespace genai
}  // namespace ov
