
## 附录 A：精简版（仅执行步骤）

> 目标：只保留“从 0 到可验证”的最短执行路径。

### A.1 前置

1. 准备 Python 环境并安装生成/验证依赖。
2. 确认 `OpenVINO_DIR` 可用。
3. 确认 `OPENVINO_TOKENIZERS_PATH_GENAI` 指向 `libopenvino_tokenizers.so`。

### A.2 生成

1. 运行 `tools/module_genai/qwen3_5_codegen.py`。
2. 输入 `py_ref/modeling_qwen3_5.py`。
3. 输出目录使用 `tools/module_genai/generated/qwen3_5_step2/`（或自定义目录）。

### A.3 产物检查

确认以下文件存在：

1. `analysis_report.json`
2. `config.generated.gpu.yaml`
3. `config.generated.cpu_compare.yaml`
4. `modules/md_generated_*.hpp/.cpp`
5. `python_ref/qwen3_5_reference_bridge.py`
6. `python_ref/compare_pipeline_with_reference.py`

### A.4 同步与构建

1. 将 `md_generated_*` 同步到 `src/cpp/src/module_genai/modules/generated`。
2. 构建工程（建议 Release）。
3. 验证 Python 可导入 `openvino_genai.ModulePipeline`。

### A.5 运行三类验证

按顺序运行：

1. text case
2. image case
3. mixed case

检查 `tools/module_genai/generated/qwen3_5_step3_reports/summary.json`：

1. `all_pass == true`
2. 每个 case `returncode == 0`

### A.6 结果判定

1. 当前允许 fallback：`fallback_used` 可为 `true`。
2. 回归目标：补齐 OpenVINO 资产后，`fallback_used == false` 且三类 case 持续通过。

### A.7 最小交付

1. 更新后的 skill 文档。
2. 生成产物目录（含 report）。
3. 一份三类样例汇总结果（`summary.json`）。
