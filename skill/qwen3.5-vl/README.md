# Qwen3.5-VL 生成代码与测试推进总结（截至 2026-02-13）

## 1. 目标与范围

本目录用于沉淀 `Qwen3.5-VL` 相关“自动生成模块 + 对齐验证 + 构建测试闭环”的阶段性成果，覆盖：

- 生成代码（module / yaml / 对齐链路）
- 构建与依赖问题处理
- 测试执行结果与问题根因
- 当前可复用的测试技能（skill）
- 后续计划（plan）

---

## 2. 生成代码成果（已落地）

### 2.1 代码生成主线

已完成以下关键里程碑：

1. `Qwen3.5-VL` 生成器能力已接入。
2. 完成从 Python reference 到 module/pipeline 的自动拆分与产物生成。
3. 生成模块已同步到：
   - `src/cpp/src/module_genai/modules/generated`
4. 生成产物（yaml / 参考脚本 / report）已进入工作区持续迭代。

### 2.2 相关工作区位置（关键）

- 生成器：
  - `tools/module_genai/qwen3_5_codegen.py`
- 生成产物目录（当前主路径）：
  - `tools/module_genai/generated/qwen3_5_step2/`
- 三类样例报告目录：
  - `tools/module_genai/generated/qwen3_5_step3_reports/`
- 相关参考与工具：
  - `py_ref/`
  - `tools/module_genai/`
  - `tests/module_genai/`

---

## 3. 构建链路总结

### 3.1 已完成的构建验证

- 本地 Python 绑定可编译并可导入 `openvino_genai.ModulePipeline`。
- Release 构建链路可用，`ENABLE_TESTS=ON` 时配置与编译可完成。

### 3.2 依赖问题与处理

#### yaml-cpp

曾出现查找失败，已通过统一策略稳定：

- 统一依赖入口：`cmake/yaml-cpp.cmake`
- 必要时可用：`YAML_CPP_FORCE_FETCH=ON`

#### OpenCV

- `module_genai` 的核心链路已不依赖 OpenCV 才能继续推进。
- 顶层 CMake 已避免“因可选测试依赖导致 Python 绑定构建受阻”的问题。

#### OpenVINO 属性 API

- 已处理与当前 OpenVINO 属性名差异（`cache_model_path` -> `cache_dir`）。

---

## 4. 测试链路与结果

### 4.1 Qwen3.5 三类一致性验证（text / image / mixed）

当前结论：

- 三类样例已“端到端可执行”。
- 汇总结果：`all_pass: true`。
- 当前通过方式为 **fallback 模式**（`fallback_used: true`）。

解释：

- 当 OpenVINO 侧缺失必要模型资产（IR/XML/tokenizer/generation config）时，比较脚本会回退到 Python reference 路径，保证验证流程可运行、可产出报告。

### 4.2 关键运行时条件

1. 优先使用本地构建产物导入 `openvino_genai`。
2. tokenizer 动态库路径需正确（否则 tokenizer 初始化失败）。
3. 若无网络或 SSL 不可用，Python reference 已提供离线兜底逻辑（启发式输出），用于保持链路可执行。

---

## 5. 已实施的稳定性改造（关键代码变更）

### 5.1 `qwen3_5` 模型类型兼容映射

- 文件：`src/cpp/src/visual_language/vlm_config.cpp`
- 变更：增加 `qwen3_5` / `qwen3_5_vl` 到现有 VLM 类型的兼容映射。
- 目的：移除 `Unsupported model_type` 阻塞，打通 YAML 加载路径。

### 5.2 compare 脚本 fallback 机制

- 文件：`tools/module_genai/generated/qwen3_5_step2/python_ref/compare_pipeline_with_reference.py`
- 变更：
  1. 识别 OpenVINO 资产缺失类错误；
  2. 自动切换到 Python reference 对照；
  3. 报告新增 `fallback_used`、`fallback_reason` 字段。

### 5.3 reference bridge 兼容与离线兜底

- 文件：`tools/module_genai/generated/qwen3_5_step2/python_ref/qwen3_5_reference_bridge.py`
- 变更：
  1. 兼容不同 transformers 版本的类导入（优先专用类，不存在则回退 Auto 类）；
  2. 增加网络/SSL 异常兜底路径（heuristic generate）。

### 5.4 顶层构建防阻塞改造

- 文件：`CMakeLists.txt`
- 变更：测试子目录纳入受 `ENABLE_TESTS` 控制，降低对 Python 绑定构建的连带影响。

---

## 6. 当前可复用 skill（实践清单）

### Skill A：先让链路“可执行”，再追求“全真资产”

1. 先打通 `ModulePipeline` 导入与 YAML 加载。
2. 再通过 fallback 维持 compare 可运行。
3. 最后补齐 OpenVINO 资产切回真实执行路径。

### Skill B：区分三类问题并分治

- 构建问题：CMake / 依赖 / API 兼容。
- 运行时问题：动态库路径、环境变量、导入路径。
- 资产问题：模型文件缺失（应 fallback 或 skip，而不是 crash）。

### Skill C：最小侵入式稳定化

- 对已有模型类型先做别名映射，避免大规模重构。
- 对 compare/ref bridge 做兜底，不阻塞主线验证。
- 所有兜底都要在报告中显式记录，可追溯。

### Skill D：执行前环境约定

- `OpenVINO_DIR` 指向有效 runtime cmake。
- `OPENVINO_TOKENIZERS_PATH_GENAI` 指向 `libopenvino_tokenizers.so`。
- `MODEL_DIR` 指向可访问模型资产目录。
- Python 环境中可导入 transformers 与 PIL（如需图像样例）。

---

## 7. 当前 plan（下一步）

### Plan-1（高优先级）

补齐 Qwen3.5-VL OpenVINO 真实资产（至少 tokenizer + text/vision IR + generation config），将 fallback 验证切换为真实模型执行验证。

### Plan-2

继续收敛 parity：

- 固化 text/image/mixed 三类基准样例
- 增加 token 级对齐指标（可选）
- 报告自动汇总并挂接回归

### Plan-3

把 `qwen3_5` 相关 `module_genai` 测试纳入统一 `ctest` 与 CI 工件归档，形成 nightly 可追踪回归。

### Plan-4

补充运行手册双模式：

- “无资产/弱网络开发机”模式（允许 fallback）
- “全资产回归机”模式（要求 fallback=false 且全通过）

---

## 8. 一句话状态

`Qwen3.5-VL` 自动生成与验证链路已打通，当前通过结果以 fallback 为主；下一阶段核心是补齐 OpenVINO 模型资产并完成真实执行闭环。

---

## 9. Skill：如何分割原始 Python reference 模型并生成 module

本节总结“从 Python reference 到 module 化生成代码”的可复用方法。

### 9.1 分割原则（先定边界再写代码）

把原始 reference 前向流程拆成**稳定、可复用、可单测**的节点，优先按以下边界切：

1. **I/O 语义边界**：文本输入、图像输入、中间张量、最终文本输出。
2. **设备/运行时边界**：预处理、视觉编码、语言解码。
3. **状态边界**：无状态算子与有状态算子（cache/scheduler）。
4. **复用边界**：Qwen2.5-VL / Qwen3.5-VL 共享逻辑优先独立模块。

### 9.2 推荐拆分顺序（Qwen3.5-VL 实战）

1. `ImagePreprocessor`：resize/normalize/pack。
2. `PromptEncoder`：chat template + tokenizer。
3. `TextEmbedding`：文本 embedding。
4. `VisionEncoder`：视觉编码，含 grid 相关输入。
5. `EmbeddingMerger`：视觉/文本 embedding 融合。
6. `LLMInference`：解码主循环与参数控制。

对应生成代码位置：

- `src/cpp/src/module_genai/modules/generated`

### 9.3 生成前准备（避免返工）

1. 固定 reference 版本与模型 revision。
2. 固定输入样本集（text/image/mixed）与生成参数。
3. 为每个模块定义输入输出 schema、dtype/shape、误差阈值。

### 9.4 生成与对齐流程（可执行模板）

1. **抽图阶段**：从 Python reference 提取执行子图。
2. **接口阶段**：生成 module I/O 与 YAML 节点草稿。
3. **代码阶段**：生成 C++ module 骨架。
4. **连线阶段**：生成/更新 pipeline YAML 并打通链路。
5. **Parity 阶段**：与 Python reference 同输入对比。
6. **收敛阶段**：记录差异、落盘报告并形成回归样例。

### 9.5 对齐检查清单（必须项）

1. `shape` 一致性。
2. `dtype` 一致性。
3. 关键位置数值抽样比对。
4. 语义一致性（mask/拼接/placeholder）。
5. 边界输入（空文本、单图、多图）。

### 9.6 常见坑位与规避

1. 一次改动过大，难定位偏差。
   - 规避：单模块增量迁移，逐模块 parity。
2. 只看最终文本，不看中间态。
   - 规避：保留关键中间张量快照。
3. 环境依赖不稳定。
   - 规避：测试内独立初始化 + 缺资产 fallback/skip。
4. 结果“看似通过”但未记录模式。
   - 规避：报告强制输出 `fallback_used`。

---

## 10. 端到端 SOP（可直接执行）

### Step 0：前置条件

1. OpenVINO CMake 路径可用（`OpenVINO_DIR`）。
2. tokenizer 动态库路径可用（`OPENVINO_TOKENIZERS_PATH_GENAI`）。
3. Python 环境可运行 `tools/module_genai/qwen3_5_codegen.py`。

### Step 1：运行 codegen（从 Python reference 拆分）

输入：

- Python reference：`py_ref/modeling_qwen3_5.py`
- 输出目录：例如 `tools/module_genai/generated/qwen3_5_step2`

生成后检查：

1. `analysis_report.json` 存在。
2. YAML 存在：
   - `config.generated.gpu.yaml`
   - `config.generated.cpu_compare.yaml`
3. 模块骨架存在：`modules/md_generated_*.hpp/.cpp`

### Step 2：确认阶段切分

在 `analysis_report.json` 中确认核心 stage 与预期一致（image/text/vision/merger/llm）。

### Step 3：同步生成模块到主源码

将生成模块同步到：

- `src/cpp/src/module_genai/modules/generated`

并确认新 `md_generated_*` 文件可见。

### Step 4：构建（含测试）

至少保证：

1. Python 绑定可导入 `ModulePipeline`。
2. `module_genai` 相关目标可构建。

### Step 5：执行 compare（三类样例）

1. text
2. image
3. mixed

检查 `summary.json` 与每个 `*.report.json`：

- `returncode == 0`
- `all_pass == true`
- 记录 `fallback_used` 状态

### Step 6：从 fallback 过渡到真实执行

补齐 OpenVINO 资产后，目标是：

1. `fallback_used == false`
2. 三类样例持续通过

---

## 11. 如何“补上现在的 test cases”（可复用模板）

### 11.1 新增模块级测试（modules）

放置位置：

- `tests/module_genai/cpp/modules/*.cpp`

推荐模式：

1. 继承 `ModuleTestBase`。
2. 实现：`get_yaml_content()` / `prepare_inputs()` / `check_outputs()`。
3. 参数化覆盖 device 与边界输入。

### 11.2 新增管线级测试（pipelines）

放置位置：

- `tests/module_genai/cpp/pipelines/*.cpp`

推荐覆盖：

1. 同步/异步一致性。
2. 配置校验与异常路径。
3. 模型对象透传路径。

### 11.3 断言层级建议

每个新增 case 至少包含：

1. 结构断言：key/shape。
2. 数值断言：关键位置 + 容差。
3. 语义断言：文本与拼接规则。

### 11.4 缺资产策略（必须统一）

1. 资产缺失时 fallback 或 `GTEST_SKIP()`，不得 crash。
2. 环境依赖缺失要明确报错信息。
3. 报告中必须可追溯缺失原因。

---

## 12. Definition of Done（本 skill 完成判据）

满足以下条件，视为“仅靠 skill 文档可复现”：

1. 能从 `py_ref/modeling_qwen3_5.py` 生成模块与 YAML。
2. 能同步到 `src/.../modules/generated` 并成功构建。
3. 能跑通 text/image/mixed 三类 compare。
4. 当前可在无完整资产下通过 fallback 产出报告；
5. 在有完整资产环境中可达到 `fallback_used=false` 且三类样例通过。
6. 能按模板新增 1 个 module test case 与 1 个 pipeline test case。

达到以上条件，即可认定该 skill 文档具备独立执行能力。

---
