# Qwen2.5-VL 生成代码与测试推进总结（截至 2026-02-13）

## 1. 目标与范围

本目录用于沉淀 `Qwen2.5-VL` 相关“自动生成模块 + 对齐验证 + 构建测试闭环”的阶段性成果，覆盖：

- 生成代码（module / yaml / 对齐链路）
- 构建与依赖问题处理
- 测试执行结果与问题根因
- 当前可复用的测试技能（skill）
- 后续计划（plan）

---

## 2. 生成代码成果（已落地）

### 2.1 代码生成主线

已完成以下关键里程碑：

1. `Qwen2.5-VL` 生成器能力已接入。
2. 完成 6 阶段 parity transform（用于生成结果与现有实现对齐）。
3. 生成模块已同步到：
   - `src/cpp/src/module_genai/modules/generated`
4. 生成产物（包括 yaml / 参考脚本 / report）已进入工作区持续迭代。

### 2.2 相关工作区位置（关键）

- 生成模块目录：
  - `src/cpp/src/module_genai/modules/generated`
- 相关参考与工具（已用于对齐验证）：
  - `py_ref/`
  - `tools/module_genai/`
  - `tests/module_genai/`

---

## 3. 构建链路总结

### 3.1 已完成的构建验证

- 56 核并行构建已多轮通过（Release）。
- 启用测试后（`ENABLE_TESTS=ON`）配置与编译均可完成。

### 3.2 依赖问题与处理

#### yaml-cpp

曾出现查找失败，已通过统一策略稳定：

- 走统一依赖入口：`cmake/yaml-cpp.cmake`
- 必要时强制抓取：`YAML_CPP_FORCE_FETCH=ON`

#### OpenCV

- 系统已安装 `libopencv-dev`，可被系统发现。
- 后续测试链路已经不再依赖 OpenCV 才能构建 `module_genai` C++ tests（见第 5 节变更）。

---

## 4. 测试链路与结果

### 4.1 continuous batching（tests/cpp）

#### 已完成

- `tests_continuous_batching` 构建通过。
- `ctest` 测试发现已打通（由 0 变为可发现并执行）。
- 268 个测试在正确环境变量下通过。

#### 关键运行时条件

- 必须设置 tokenizer 动态库环境变量：
  - `OPENVINO_TOKENIZERS_PATH_GENAI`
- 根因经验：缺失该变量会导致 `AddSecondInputTest` 等 tokenizer 相关测试失败。

### 4.2 module_genai C++ tests（tests/module_genai/cpp）

当前运行结果：

- 总计：83
- 通过：30
- 失败：0
- 跳过：53

说明：

- 跳过主要来自大模型资产缺失（`test_models` 目录下模型未就绪），而非构建/框架故障。

---

## 5. 已实施的测试稳定性改造（关键代码变更）

### 5.1 CTest/GTest 发现链路

- 在 `tests/cpp/CMakeLists.txt` 中增加了 gtest 自动发现注册。
- 在顶层 `CMakeLists.txt` 中完成 testing 初始化，使 `ctest` 可管理测试。

### 5.2 tokenizer 初始化顺序依赖修复

- 文件：`tests/cpp/test_add_second_input_pass.cpp`
- 修复点：在 `run_add_second_input_pass()` 中确保 shared tokenizer 在测试内独立初始化。
- 效果：避免测试依赖“先后执行顺序”导致的偶发失败。

### 5.3 module_genai 对 OpenCV 的硬依赖解除

- 文件：`tests/module_genai/cpp/CMakeLists.txt`
- 变更：去除 OpenCV `find_package` 强依赖与链接。

- 文件：`tests/module_genai/cpp/utils/load_image.cpp`
- 变更：`create_countdown_frames()` 改为纯 Tensor 填充实现，不再使用 OpenCV API。

- 文件：`CMakeLists.txt`
- 变更：`ENABLE_TESTS=ON` 时直接纳入 `tests/module_genai/cpp`，不再因 OpenCV 缺失跳过整个子目录。

### 5.4 缺失模型资产时“可控跳过”

为避免 model-less 环境下崩溃/硬失败，已增加显式 `GTEST_SKIP()`：

- `tests/module_genai/cpp/utils/ut_modules_base.hpp`
- `tests/module_genai/cpp/pipelines/generate_async_acc.cpp`
- `tests/module_genai/cpp/pipelines/modulepipeline_pass_ovmodel.cpp`

---

## 6. 当前可复用 skill（实践清单）

### Skill A：先打通“可发现测试”再谈“测试通过率”

1. 确保顶层 testing 初始化。
2. 确保 gtest 测试注册到 ctest。
3. 先 `ctest -N` 验证发现数量，再执行。

### Skill B：区分“构建问题 / 运行时问题 / 资产问题”

- 构建问题：依赖或 CMake 配置。
- 运行时问题：动态库路径、环境变量。
- 资产问题：模型文件缺失导致需要 skip 或补齐模型。

### Skill C：优先做“最小侵入式稳定化”

- 针对单测顺序依赖，做测试内初始化兜底。
- 对可选依赖（OpenCV）优先降耦，避免阻塞全链路。
- 对缺失资源用 `GTEST_SKIP()`，保持 CI 可执行与结果可解释。

### Skill D：执行前置环境约定

- `OpenVINO_DIR` 指向有效 runtime cmake。
- `OPENVINO_TOKENIZERS_PATH_GENAI` 指向 `libopenvino_tokenizers.so`。
- `MODEL_DIR` / `DATA_DIR` 明确指向测试资产目录。

---

## 7. 当前 plan（下一步）

### Plan-1（高优先级）

补齐 `tests/module_genai/cpp/test_models` 所需模型资产，减少 53 个 skip，转为实测。

### Plan-2

对 `Qwen2.5-VL` 自动生成模块继续做 parity 回归：

- 生成 YAML 与手工实现对齐
- Python reference 对比误差门限固化
- 回归报告自动化输出

### Plan-3

将 module_genai tests 纳入统一 `ctest` 调度与 CI 工件归档，形成稳定 nightly 回归。

### Plan-4

补充文档化运行手册：

- “无模型资产（开发机）”模式：允许 skip、验证框架稳定性
- “有模型资产（回归机）”模式：要求全量通过

---

## 8. 一句话状态

`Qwen2.5-VL` 自动生成与构建/测试链路已打通；当前瓶颈已从“工程能力问题”收敛为“模型测试资产就绪度问题”。

---

## 9. Skill：如何分割原始 Python reference 模型并生成 module

本节总结“从 Python reference 到 module 化生成代码”的可复用方法。

### 9.1 分割原则（先定边界再写代码）

把原始 reference 前向流程拆成**稳定、可复用、可单测**的节点，优先按以下边界切：

1. **I/O 语义边界**：文本输入、图像输入、张量中间态、最终文本输出。
2. **设备/运行时边界**：CPU 预处理、NPU/GPU 编码器、LLM 解码器。
3. **状态边界**：无状态算子（易并行）与有状态算子（如 cache / scheduler）。
4. **复用边界**：在 Qwen2.5-VL / Qwen3.5 / 其它多模态管线可共享的逻辑优先独立模块。

### 9.2 推荐拆分顺序（Qwen2.5-VL 实战）

建议按“输入到输出”顺序递进拆分，避免一次改太大：

1. `ImagePreprocessor`：把 Python 中 resize/normalize/pack 逻辑固化。
2. `VisionEncoder`：把视觉塔前向独立出来，仅保留必需输入输出。
3. `EmbeddingMerger`：对齐多模态拼接规则（位置、mask、type ids）。
4. `LLMInference`：文本解码主循环与采样参数分离。
5. `PostProcess`：输出整理（token → text / structured output）。

对应生成代码位置：

- `src/cpp/src/module_genai/modules/generated`

### 9.3 生成前准备（避免返工）

1. 固定 reference 版本与权重版本（commit + model revision）。
2. 固定输入样本集（图像、prompt、seed、generation config）。
3. 为每个候选模块定义：
  - 输入 schema
  - 输出 schema
  - dtype / shape 约束
  - 允许误差（数值阈值）

### 9.4 生成与对齐流程（6 阶段可执行模板）

1. **抽图阶段**：从 Python reference 提取执行子图和依赖关系。
2. **接口阶段**：生成 module 输入输出定义与 YAML 节点草稿。
3. **代码阶段**：生成 C++ module 实现（先最小可运行）。
4. **连线阶段**：生成/更新 pipeline YAML，打通端到端链路。
5. **Parity 阶段**：同输入对比 Python reference 输出（shape + 数值 + 语义）。
6. **收敛阶段**：将差异收敛到可解释范围，沉淀报告与回归样例。

### 9.5 对齐检查清单（必须项）

每个模块至少验证：

1. `shape` 一致性（含 batch、seq、hidden 维度）。
2. `dtype` 一致性（含 int/float 混用路径）。
3. 前 N 项数值比对（用于快速定位偏差）。
4. 关键语义比对（例如 token 拼接位置、mask 语义）。
5. 边界输入（空文本、单图、多图、极端长度）。

### 9.6 常见坑位与规避

1. **把 reference 辅助逻辑误当核心算子**：导致模块职责混乱。  
  - 规避：先按 I/O 语义画 DAG，再决定模块边界。

2. **一次性大改**：很难定位偏差来源。  
  - 规避：单模块增量迁移，逐模块 parity。

3. **只看最终文本，不看中间张量**：问题定位成本高。  
  - 规避：保留关键中间张量快照与 compare 脚本。

4. **测试依赖顺序或外部环境**：结果不稳定。  
  - 规避：每个测试独立初始化运行时依赖；缺资产显式 skip。

### 9.7 产物落盘规范（建议）

每次模块化迭代都应落盘：

1. 生成代码路径与变更说明。
2. 对应 YAML（pipeline）变更。
3. parity 对比报告（含失败样例）。
4. 可复现实验参数（seed、prompt、模型版本、设备）。

这样可以保证后续在 `Qwen2.5-VL` 之外复用同一套 module 生成方法。

---

## 10. 结论：README 是否“单独可执行”与补足项

### 10.1 检查结论

仅靠本文件**此前版本**，还不能 100% 复现“从 Python reference 拆到当前 modules，并补上现有测试用例”的全流程。  
缺口主要在：

1. 缺少**一步一步可执行 SOP**（含命令顺序与产物检查点）。
2. 缺少“**如何把生成模块同步到 src 并参与构建**”的收敛路径细节。
3. 缺少“**如何新增/扩展 module test case**”的模板化方法。
4. 缺少“**模型资产就绪门槛**”与 skip/pass 的判定标准。

本节已将以上缺口补齐。

### 10.2 端到端 SOP（可直接执行）

#### Step 0：前置条件

1. OpenVINO CMake 路径可用（`OpenVINO_DIR`）。
2. Qwen2.5-VL 模型资产放在 `MODEL_DIR`（见 Step 6）。
3. tokenizer 动态库可访问（`OPENVINO_TOKENIZERS_PATH_GENAI`）。

#### Step 1：运行 codegen（从 Python reference 拆分）

使用：

- `tools/module_genai/qwen2_5_vl_codegen.py`

输入：

- Python reference：`py_ref/modeling_qwen2_5_vl.py`
- 输出目录：`tools/module_genai/generated_qwen2_5_vl`

生成后必须检查：

1. `analysis_report.json` 存在且 `effective_spec` 合法。
2. 生成 YAML：
  - `config.generated.gpu.yaml`
  - `config.generated.cpu_compare.yaml`
3. 生成模块骨架：`modules/md_generated_*.hpp/.cpp`

#### Step 2：确认阶段切分正确

在 `analysis_report.json` 中确认：

1. `xml_gpu_stages` 包含：`image_preprocessor` / `text_embedding` / `vision_encoder` / `llm_inference`。
2. `local_logic_stages` 包含：`prompt_encoder` / `embedding_merger`。

若与预期不一致，优先修正 spec 再重跑 codegen，不要手改大量生成产物。

#### Step 3：同步生成模块到主源码

执行生成目录下脚本：

- `tools/module_genai/generated_qwen2_5_vl/sync_generated_modules_to_src.sh`

同步后检查目标目录：

- `src/cpp/src/module_genai/modules/generated`

需出现最新 `md_generated_*` 文件。

#### Step 4：构建（含测试）

使用统一 CMake 配置（Release + `ENABLE_TESTS=ON`）。

最少应完成：

1. `tests_continuous_batching` 可构建。
2. `genai_modules_test` 可构建。

#### Step 5：先跑通基础测试闭环

1. `tests_continuous_batching`：
  - 要求 `OPENVINO_TOKENIZERS_PATH_GENAI` 正确。
2. `genai_modules_test`：
  - 无模型资产时允许 skip，但不能 crash/fail。

#### Step 6：模型资产门槛（把 skip 转为真实验证）

`tests/module_genai/cpp/test_models` 至少应包含 Qwen2.5-VL 所需 XML 资产，例如：

1. `openvino_text_embeddings_model.xml`
2. `openvino_vision_embeddings_model.xml`
3. 其他 pipeline 所需 IR/XML/配置文件

资产就绪后再次运行 `genai_modules_test`，目标是减少 skip 并提升真实通过数。

---

## 11. 如何“补上现在的 test cases”（可复用模板）

### 11.1 新增模块级测试（modules）

放置位置：

- `tests/module_genai/cpp/modules/*.cpp`

推荐模式：

1. 继承 `ModuleTestBase`。
2. 实现三件事：
  - `get_yaml_content()`：构建最小可测 pipeline。
  - `prepare_inputs()`：准备输入。
  - `check_outputs()`：做 shape/数值断言。
3. 用参数化测试覆盖：
  - device（CPU/GPU）
  - 典型输入与边界输入

现成参考：

1. `VisionEncoderModule.cpp`
2. `ImagePreprocesModule.cpp`
3. `LLMInferenceModule.cpp`

### 11.2 新增管线级测试（pipelines）

放置位置：

- `tests/module_genai/cpp/pipelines/*.cpp`

推荐覆盖：

1. 同步/异步一致性（参考 `generate_async_acc.cpp`）。
2. 配置校验与异常路径（参考 `module_pipeline_validator.cpp`）。
3. 模型对象透传路径（参考 `modulepipeline_pass_ovmodel.cpp`）。

### 11.3 断言层级建议

每个新增 case 至少包含：

1. 结构断言：输出 key 存在、shape 正确。
2. 数值断言：关键前 N 项 + 容差阈值。
3. 语义断言：文本/拼接规则/placeholder 一致。

### 11.4 缺资产策略（必须统一）

原则：

1. 资产缺失时 `GTEST_SKIP()`，不要抛未处理异常。
2. 环境依赖缺失时输出明确提示（缺哪个文件/变量）。
3. 不允许因为缺资产导致进程崩溃（segfault）。

---

## 12. Definition of Done（本 skill 完成判据）

满足以下条件，视为“仅靠 skill 文档可复现”：

1. 能从 `py_ref/modeling_qwen2_5_vl.py` 生成模块与 YAML。
2. 能同步到 `src/.../modules/generated` 并成功构建。
3. 能跑通 `tests_continuous_batching`（环境变量正确时全通过）。
4. 能跑通 `genai_modules_test`（至少 0 fail；缺资产允许 skip）。
5. 能按模板新增 1 个 module test case 与 1 个 pipeline test case。

达到以上 5 条，即可认定该 skill 文档已具备独立执行能力。
