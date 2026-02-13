# Qwen3.5 从 Python 原始实现到 Module + YAML Pipeline 计划

## 1. 目标与范围

目标：参考现有 Qwen2.5-VL 方案（skill + codegen + README），把 `py_ref/modeling_qwen3_5.py` 拆解为可执行的 Module GenAI 模块与 YAML pipeline，并在 CPU 上验证与 Python 原始实现一致性。

范围包含：
- 自动分析与拆分（stage mapping）
- YAML 生成（GPU 运行版 + CPU 对比版）
- 模块骨架生成（可先桥接 Python 原始逻辑）
- 参考桥接脚本与一致性对比脚本
- 最小可复现验证样例（CPU）

不在首轮范围：
- 首轮不强制实现所有模块的高性能 C++ 等价算子
- 首轮不做跨设备性能最优，仅保证可对齐与可验证

---

## 2. 现状与关键前置问题

已识别风险：`py_ref/modeling_qwen3_5.py` 当前是 GitHub HTML 页面快照，不是可直接 AST 解析的 Python 源文件。

前置修复：
1. 增加“HTML 兜底提取器”：从 HTML 内嵌 JSON 的 `rawLines` 提取真实 Python 内容并落盘。
2. 规范源文件输入优先级：
   - 优先真实 `.py`
   - 其次 HTML 内嵌 `rawLines` 自动恢复
   - 失败时给出明确报错与修复提示

完成标准：`modeling_qwen3_5.py` 可以被解析并输出稳定的 class/function/signals 报告。

---

## 3. Qwen3.5 拆分策略（stage mapping）

基于 `Qwen3_5ForConditionalGeneration` 主路径，拆分如下：

1. `ParameterModule`
- 输入：`image`、`prompt`（后续可扩展 `video`）

2. `ImagePreprocessModule`
- 职责：图像预处理、构造视觉输入张量

3. `TextEncoderModule`
- 职责：chat template + tokenizer，产出 `input_ids`、`attention_mask`

4. `TextEmbeddingModule`
- 职责：`get_input_embeddings` / token embedding

5. `VisionEncoderModule`
- 职责：视觉编码（含 `image_grid_thw` / `video_grid_thw`、vision 特征）
- 输出：`image_embedding`、`video_embedding`、`position_ids`、`rope_delta`

6. `EmbeddingMergerModule`
- 职责：将视觉 embedding 按 placeholder/mask 融入文本 embedding

7. `LLMInferenceModule`
- 职责：language model forward + generate（首轮 CPU 对齐优先）
- 关键：对齐 `rope_deltas`、cache 行为、decode 参数

8. `ResultModule`
- 输出：`generated_text`

建议拓扑：
`pipeline_params -> image_preprocessor -> prompt_encoder -> text_embedding -> vision_encoder -> embedding_merger -> llm_inference -> pipeline_result`

---

## 4. 与 Qwen2.5-VL 共用与差异化实现

可复用项：
- codegen 脚手架结构（YAML 生成、模块骨架、bridge、compare）
- CPU compare 流程（ModulePipeline vs Python baseline）
- 设备策略（hybrid/all-gpu/all-cpu）

Qwen3.5 需要新增或加强：
- 对 `Qwen3_5ForConditionalGeneration` 的类名与信号识别
- `get_rope_index` / `rope_deltas` 对齐
- 视频路径信号（`pixel_values_videos`、`video_grid_thw`）预留
- 线性注意力相关 cache 信号识别（用于风险提示与对齐选项）

---

## 5. 实施步骤（分阶段）

### 阶段 A：代码生成器扩展
1. 新建 `tools/module_genai/qwen3_5_codegen.py`（建议独立，不破坏 qwen2_5）
2. 抽象公共逻辑到共享函数（可后续再抽）
3. 加入 HTML->rawLines 源码恢复能力
4. 生成物与 qwen2_5 对齐：
   - `config.generated.gpu.yaml`
   - `config.generated.cpu_compare.yaml`
   - `modules/md_generated_*.hpp/.cpp`
   - `python_ref/qwen3_5_reference_bridge.py`
   - `python_ref/compare_pipeline_with_reference.py`
   - `analysis_report.json`

### 阶段 B：Skill 与文档
1. 新增 `.gnai/skills/model-to-module-pipeline-qwen3_5.md`
2. 更新 `tools/module_genai/README.md`，加入 qwen3_5 使用说明
3. 明确“原始 Python 片段复用”路径：
   - 通过 bridge 在模块 `run_with_python_reference()` 中调用

### 阶段 C：CPU 一致性验证
1. 固定 decode 参数（建议 `do_sample=false`、固定 `max_new_tokens`）
2. 统一输入样例（固定 prompt + image）
3. 输出对比维度：
   - 文本精确匹配
   - 归一化后包含匹配
   - 可选：首 token logits/top-k 一致率
4. 产出 JSON 报告，包含 `pass/fail` 与误差摘要

### 阶段 D：回归与验收
1. 至少 3 组样例回归（短问答、描述、OCR 风格）
2. YAML 拓扑与端口完整性检查（无悬空 source）
3. 在 CPU-only 环境稳定运行

---

## 6. CPU 一致性验收标准（DoD）

必达：
- `config.generated.cpu_compare.yaml` 可被 `ModulePipeline` 正常加载
- 与 Python baseline 对比报告中至少满足：
  - `exact_match=true` 或
  - `contains_match=true`
- 三组固定样例通过率 >= 90%

建议增强：
- 增加 token-level 前 N token 一致率指标
- 增加失败 case 自动保存（输入、两侧输出、中间特征摘要）

---

## 7. 风险与应对

1. 源文件是 HTML 导致无法解析
- 应对：实现 HTML `rawLines` 自动恢复

2. 采样导致文本不稳定
- 应对：CPU 对比统一关闭采样，固定生成参数

3. 线性注意力/cache 路径差异造成偏差
- 应对：首轮聚焦短序列、无复杂 cache 场景；在报告中标记 cache 模式

4. 视频分支暂未完整验证
- 应对：先保留端口与参数，图像路径先闭环

---

## 8. 交付物清单

- `.gnai/skills/model-to-module-pipeline-qwen3_5.md`
- `tools/module_genai/qwen3_5_codegen.py`
- `tools/module_genai/README.md` 更新
- 生成产物目录（示例）：
  - `config.generated.gpu.yaml`
  - `config.generated.cpu_compare.yaml`
  - `modules/md_generated_*.hpp/.cpp`
  - `python_ref/qwen3_5_reference_bridge.py`
  - `python_ref/compare_pipeline_with_reference.py`
  - `analysis_report.json`
  - `compare_report.json`

---

## 9. 建议执行顺序（最短路径）

1. 先做 HTML 源恢复 + qwen3_5 分析报告
2. 再落地 YAML 与 bridge 脚本
3. 跑通 CPU 一致性最小样例
4. 最后补 skill 文档与 README

这样可以最早拿到“可运行 + 可对比 + 可迭代”的版本。