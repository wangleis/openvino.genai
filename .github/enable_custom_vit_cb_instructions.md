# Extend CB for custom VIT inputs_embeds

## Description

扩展 `ContinuousBatchingPipeline`，支持客户自定义 VIT 产出的 `inputs_embeds`，并传入 deepstack 相关额外输入。

当前已验证可运行的方案：

- 输入主干：`inputs_embeds` + `position_ids`
- 额外输入仅使用：
	- `deepstack_visual_embeds`
	- `visual_pos_masks`
- 对于 full-seq deepstack（`[L, S, H]`），先基于 `visual_pos_mask` 压缩为稀疏 deepstack（`[L, V, H]`）再传给 CB。

## Build

```
cd ~/mygithub/modular_genai/composable_pipeline/thirdparty/openvino.genai
./build.sh
```

## Run sample

```
cd ~/mygithub/modular_genai/composable_pipeline/thirdparty/openvino.genai
./run_sample.sh
```

输出文本可读即可，语义与 sample 内 `expected_text` 类似即可。

## Implementation summary (current)

### 1) Dummy ModelingVL classes
当 VIT 已由客户处理、直接传递 `inputs_embeds` 时，仍然走 VLM pipeline。提供 dummy 版本以满足接口：

- [src/cpp/src/visual_language/modeling_vl/classes.hpp](src/cpp/src/visual_language/modeling_vl/classes.hpp)
- [src/cpp/src/visual_language/modeling_vl/classes.cpp](src/cpp/src/visual_language/modeling_vl/classes.cpp)

### 2) ModelingVL model_type 支持
新增 `model_type: "modeling_vl"` 以绕开已有模型类型约束：

- `VLMModelType` 新增 `MODELING_VL`
- `vlm_config.cpp` 中加入 `{"modeling_vl", VLMModelType::MODELING_VL}`

### 3) VisionEncoder/InputsEmbedder 选择
支持 `MODELING_VL`：

- [src/cpp/src/visual_language/vision_encoder.cpp](src/cpp/src/visual_language/vision_encoder.cpp)
- [src/cpp/src/visual_language/inputs_embedder.cpp](src/cpp/src/visual_language/inputs_embedder.cpp)

### 4) Sample 关键点
`samples/cpp/custom_vit_cb/custom_vit_cb.cpp` 中通过 `generate()` 传入：

- `inputs_embeds`
- `position_ids`（可选）
- `extra_inputs`：
	- `deepstack_visual_embeds`（推荐稀疏布局 `[L, V, H]`）
	- `visual_pos_masks`

补充：

- sample 默认 `step=0`（若不存在则回退到最新 step）。
- `SchedulerConfig.max_num_batched_tokens = 1` 为当前稳定配置（已验证可跑通 prefill + generate）。

不建议在该 sample 中把 `attention_mask` / `beam_idx` 作为 `extra_inputs` 透传。

### 5) model_runner deepstack 处理说明（已对齐）

- `model_runner` 已支持通过 `visual_pos_mask(s)` 获取视觉位置并填充 deepstack。
- deepstack 在 CB 内部按 dense 布局组织并参与后续推理。
- 输入名兼容 `visual_pos_mask` 与 `visual_pos_masks`。

### 6) 配置要求
`config.json` 必须包含 `model_type: "modeling_vl"`，以避免 `Unsupported 'xxx' VLM model type`。
