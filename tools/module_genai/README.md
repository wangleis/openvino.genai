# Qwen2.5-VL 可执行生成器

脚本：
- [tools/module_genai/qwen2_5_vl_codegen.py](tools/module_genai/qwen2_5_vl_codegen.py)

## 功能
- 读取 Python 模型定义文件（如 `modeling_qwen2_5_vl.py`）
- 生成 YAML pipeline：
  - `config.generated.gpu.yaml`
  - `config.generated.cpu_compare.yaml`（可选）
- 生成模块骨架文件（`modules/md_generated_*.hpp/.cpp`）
- 生成 Python 参考桥接与 CPU 对比脚本：
  - `python_ref/qwen2_5_vl_reference_bridge.py`
  - `python_ref/compare_pipeline_with_reference.py`
- 输出 `analysis_report.json`，包含阶段切分：
  - `xml_gpu_stages`: 适合做成 XML 并在 GPU 跑
  - `local_logic_stages`: 应在模块内实现的 Python reference 逻辑/运算

## 阶段切分建议（Qwen2.5-VL）
- XML + GPU（可先由 Python reference 对齐）：
  - `image_preprocessor`
  - `text_embedding`
  - `vision_encoder`
  - `llm_inference`

- 模块内本地实现（参考现有 `md_*` 风格）：
  - `prompt_encoder`（模板、tokenizer、placeholder 展开）
  - `embedding_merger`（token 对齐替换与一致性校验）

## 示例
```bash
python tools/module_genai/qwen2_5_vl_codegen.py \
  --python-model-file py_ref/modeling_qwen2_5_vl.py \
  --model-dir ./tests/module_genai/cpp/test_models/Qwen2.5-VL-3B-Instruct/INT4/ \
  --output-dir ./tools/module_genai/generated_qwen2_5_vl \
  --skill-md ./.gnai/skills/model-to-module-pipeline-qwen2_5_vl.md \
  --tools-md ./tools/module_genai/README.md \
  --gpu-policy hybrid \
  --emit-generated-types-yaml \
  --emit-cpu-compare-yaml \
  --emit-iteration-plan \
  --emit-local-parity-reference
```

## MD 驱动优先级
- CLI 参数
- `tools-md` 中 `CODEGEN_SPEC_JSON`
- `skill-md` 中 `CODEGEN_SPEC_JSON`
- 生成器内置默认值

## 生成结果说明
- YAML 中会为各阶段写入 `params.use_python_reference`：
  - XML+GPU 阶段默认 `true`
  - 本地逻辑阶段默认 `false`
- `--emit-generated-types-yaml` 会额外生成：
  - `config.generated.generated_types.yaml`
  其中 `prompt_encoder`/`embedding_merger` 类型分别替换为
  `GeneratedPromptEncoderModule`/`GeneratedEmbeddingMergerModule`，用于验证自动生成模块。
- 本地逻辑阶段（`prompt_encoder` / `embedding_merger`）默认优先使用“现有模块实现的类名替换模板”生成，
  目标是更接近可编译、可对齐版本，而不是空 skeleton。
- 如需退回 skeleton，可加参数：`--disable-local-parity-transform`
- `--emit-local-parity-reference` 会在输出目录导出：
  - `modules/parity_existing/md_text_encoder.*`
  - `modules/parity_existing/md_embedding_merger.*`
  用于本地逻辑阶段按现有实现快速对齐。

## 编译接入（自动脚本）
生成目录内会产出：
- `sync_generated_modules_to_src.sh`

执行后会把生成模块同步到：
- `src/cpp/src/module_genai/modules/generated/`

由于 [src/cpp/CMakeLists.txt](src/cpp/CMakeLists.txt) 使用 `src/**/*.cpp` 递归收集源码，
同步后重编译即可自动纳入构建。

## CPU 对比测试
1. 先用 `config.generated.cpu_compare.yaml` 跑 ModulePipeline
2. 再用 `qwen2_5_vl_reference_bridge.py` 跑 Python 原始模型
3. 用 `compare_pipeline_with_reference.py` 输出对比报告 JSON

> 注意：GPU 侧模块骨架默认预留 `run_with_python_reference()`，用于接入 Python 原始实现（pybind11 或 subprocess 文件交换）。
> 注意：模块骨架内支持 `params.use_python_reference=true/false`，便于 GPU 上逐阶段从 reference 切换到本地实现。

## CODEGEN_SPEC_JSON

```json
{
  "spec_name": "qwen2_5_vl_codegen_spec",
  "spec_version": "1.0",
  "output_dir_default": "tools/module_genai/generated_qwen2_5_vl",
  "yaml_output_files": {
    "gpu": "config.generated.gpu.yaml",
    "cpu_compare": "config.generated.cpu_compare.yaml"
  },
  "module_generation": {
    "skeleton_prefix": "md_generated_",
    "respect_existing_style": true,
    "use_python_reference_switch": true
  },
  "parity_policy": {
    "except_xml_loading": true,
    "match_existing_module_behavior": true
  }
}
```
