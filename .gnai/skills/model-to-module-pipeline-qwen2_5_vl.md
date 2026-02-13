# Skill: 从 Python 模型自动生成 Module + YAML Pipeline（Qwen2.5-VL）

## Description
从 `modeling_qwen2_5_vl.py`（或同类 VLM Python 模型）自动抽取推理阶段，映射到 OpenVINO Module GenAI 模块，并生成：

1. 模块拆分建议（含 CPU/GPU 放置）
2. `config.yaml` pipeline
3. （可选）缺失模块的 C++ 模块骨架文件（`md_xxx.hpp/.cpp`）

本 Skill 优先对齐现有 Module GenAI 实现与测试：
- `samples/cpp/module_genai/config_yaml/Qwen2.5-VL-3B-Instruct/config.yaml`
- `tests/module_genai/cpp/pipelines/qwen2_5_vl.cpp`

## Trigger
当用户请求：
- “分析 Qwen2.5-VL 模块拆分”
- “从 Python 模型自动生成 module + yaml”
- “给出 CPU/GPU 分工并产出 pipeline”

## Inputs
- `python_model_file`: Python 模型定义文件（优先 `modeling_qwen2_5_vl.py`）
- `model_dir`: OpenVINO 导出模型目录
- `target_yaml`: 目标 YAML 输出路径
- `prefer_device`: `GPU` / `CPU` / `HYBRID`

## Stage Mapping（Python -> Module）
按如下优先级识别关键函数/逻辑：

1. 视觉预处理（`pixel_values`、grid 相关）
   - Python 信号：`get_image_features` / `get_video_features` 前的数据准备
   - Module：`ImagePreprocessModule`

2. 文本模板与分词（chat template / tokenizer）
   - Python 信号：tokenizer、placeholder token 拼接
   - Module：`TextEncoderModule`

3. 文本嵌入
   - Python 信号：`get_input_embeddings`、text embedding
   - Module：`TextEmbeddingModule`

4. 视觉编码 + 位置相关特征
   - Python 信号：vision transformer、`grid_thw`、window index、rotary vision pos
   - Module：`VisionEncoderModule`

5. 多模态 embedding 合并
   - Python 信号：placeholder mask + `masked_scatter` 或等价替换逻辑
   - Module：`EmbeddingMergerModule`

6. LLM 解码/生成
   - Python 信号：language model forward / generate / kv-cache / rope delta
   - Module：`LLMInferenceModule`

7. 输出收集
   - Module：`ResultModule`

## Device Placement Rules（默认 HYBRID）

### 推荐 GPU
- `ImagePreprocessModule`（当视觉前处理含 OpenVINO 视觉 encoder 推理）
- `TextEmbeddingModule`
- `VisionEncoderModule`
- `EmbeddingMergerModule`
- `LLMInferenceModule`

### 推荐 CPU
- `ParameterModule`
- `TextEncoderModule`（chat template + tokenizer，通常更偏 CPU 逻辑）
- `ResultModule`

## XML + Python Reference + 本地逻辑 切分（Qwen2.5-VL）

### 1) 适合转成 XML 并加载到 GPU（先用 Python reference 对齐）
- `image_preprocessor`
- `text_embedding`
- `vision_encoder`
- `llm_inference`

### 2) 不建议仅靠 Python reference，需在模块内实现（对齐现有 md_* 风格）
- `prompt_encoder`
   - chat template
   - placeholder 扩展
   - tokenizer 编码与 `images_sequence` 计算
- `embedding_merger`
   - image/video placeholder token 定位
   - embedding 替换与数量一致性校验

### 例外
- 若平台为 iGPU / dGPU 且图文批量较大，可将 `TextEncoderModule` 与 `ImagePreprocessModule` 放 GPU 以减少 Host-Device 往返。
- 若 GPU 显存受限，优先将 `ImagePreprocessModule` 留在 CPU。

## YAML 生成模板（Qwen2.5-VL）
必须包含以下拓扑：

`pipeline_params -> image_preprocessor -> prompt_encoder -> text_embedding -> vision_encoder -> embedding_merger -> llm_inference -> pipeline_result`

并保证端口一致：
- `prompt_encoder.outputs`: `input_ids`, `mask`, `images_sequence`
- `vision_encoder.outputs`: `image_embedding`, `video_embedding`, `position_ids`, `rope_delta`
- `llm_inference.inputs`: `embeds`, `position_ids`, `rope_delta`

## 自动生成步骤
1. 读取 Python 文件，抽取函数/类名与关键调用（关键词：`get_image_features`, `get_rope_index`, `masked_scatter`, `language_model`）。
2. 根据 Stage Mapping 建立 DAG。
3. 对每个 stage 套用设备规则生成模块节点。
4. 生成 `config.yaml`（含 `global_context.model_type`、`pipeline_modules`）。
5. 若遇到现有库中无对应模块类型：
   - 生成 `src/cpp/src/module_genai/modules/md_<name>.hpp`
   - 生成 `src/cpp/src/module_genai/modules/md_<name>.cpp`
   - 按 `IBaseModule` 模板填充 `initialize()` / `run()`。

## 输出约束
- 端口名必须可追溯到上游输出，禁止悬空 source。
- 参数项（如 `model_path`, `vision_start_token_id`）必须显式写入 YAML。
- 若输入模型包含视频分支但当前部署仅图片，仍保留 `video_embedding` 端口（可为空张量）。

## Qwen2.5-VL 当前结论（可直接使用）
- 现有实现的模块切分是合理的：
  - 预处理 / 编码 / 合并 / 生成已完整闭环
- 推荐运行放置：
  - GPU: `image_preprocessor`, `text_embedding`, `vision_encoder`, `embedding_merger`, `llm_inference`
  - CPU: `pipeline_params`, `prompt_encoder`, `pipeline_result`
- 兼容现有样例 YAML 与测试用例路径。

## Acceptance Checklist
- [ ] 生成的 YAML 能被 `ModulePipeline` 正常加载
- [ ] 每个模块 `inputs.source` 都存在
- [ ] `generated_text` 可从 `pipeline_result` 拿到
- [ ] CPU/GPU 放置符合规则或用户显式覆盖

## CODEGEN_SPEC_JSON

```json
{
   "spec_name": "qwen2_5_vl_codegen_spec",
   "spec_version": "1.0",
   "model_type": "qwen2_5_vl",
   "pipeline_topology": [
      "pipeline_params",
      "image_preprocessor",
      "prompt_encoder",
      "text_embedding",
      "vision_encoder",
      "embedding_merger",
      "llm_inference",
      "pipeline_result"
   ],
   "stage_split": {
      "xml_gpu_stages": [
         "image_preprocessor",
         "text_embedding",
         "vision_encoder",
         "llm_inference"
      ],
      "local_logic_stages": [
         "prompt_encoder",
         "embedding_merger"
      ]
   },
   "default_devices": {
      "pipeline_params": "CPU",
      "image_preprocessor": "GPU",
      "prompt_encoder": "CPU",
      "text_embedding": "GPU",
      "vision_encoder": "GPU",
      "embedding_merger": "GPU",
      "llm_inference": "GPU",
      "pipeline_result": "CPU"
   },
   "parity_target": {
      "match_existing_modules": true,
      "note": "除 XML 模型加载路径外，其余逻辑与现有 md_* 模块实现保持一致"
   }
}
```
