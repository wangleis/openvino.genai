#!/usr/bin/env python3
"""
Qwen2.5-VL module pipeline code generator.

功能：
1) 读取 Python 模型文件（例如 modeling_qwen2_5_vl.py）并分析关键阶段
2) 生成 Module GenAI YAML pipeline（GPU 运行版本 + CPU 对比版本）
3) 生成模块骨架文件（hpp/cpp）
4) 生成 Python 参考桥接与 CPU 对比测试脚本

设计原则：
- GPU 阶段模块骨架默认预留 `run_with_python_reference()`，用于对接 Python 原始实现
- CPU 对比模式使用 openvino_genai.ModulePipeline 与 transformers 原始模型做基线比较
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


XML_GPU_STAGES = [
    "image_preprocessor",
    "text_embedding",
    "vision_encoder",
    "llm_inference",
]

LOCAL_LOGIC_STAGES = [
    "prompt_encoder",
    "embedding_merger",
]

DEFAULT_DEVICES = {
    "pipeline_params": "CPU",
    "image_preprocessor": "GPU",
    "prompt_encoder": "CPU",
    "text_embedding": "GPU",
    "vision_encoder": "GPU",
    "embedding_merger": "GPU",
    "llm_inference": "GPU",
    "pipeline_result": "CPU",
}


@dataclass
class ModelAnalysis:
    class_names: List[str]
    function_names: List[str]
    found_signals: Dict[str, bool]
    notes: List[str]


def extract_codegen_spec_from_md(md_path: Path | None) -> Dict[str, object]:
    if md_path is None or not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"##\s+CODEGEN_SPEC_JSON\s*```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.MULTILINE)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def deep_merge_dict(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _replace_class_symbols(text: str, old_class: str, new_class: str) -> str:
    # 类名、注册宏、构造析构、日志文案统一替换
    text = re.sub(rf"\b{re.escape(old_class)}\b", new_class, text)
    return text


def _replace_include_header(text: str, old_header: str, new_header: str) -> str:
    return text.replace(f'#include "{old_header}"', f'#include "{new_header}"')


def generate_stage_from_existing(
    module_name: str,
    repo_root: Path,
    generated_header: str,
) -> Tuple[str, str] | None:
    mapping = {
        "image_preprocessor": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_img_preprocess.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_img_preprocess.cpp",
            "old_class": "ImagePreprocessModule",
            "new_class": "GeneratedImagePreprocessorModule",
            "old_header": "md_img_preprocess.hpp",
        },
        "prompt_encoder": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_text_encoder.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_text_encoder.cpp",
            "old_class": "TextEncoderModule",
            "new_class": "GeneratedPromptEncoderModule",
            "old_header": "md_text_encoder.hpp",
        },
        "text_embedding": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_text_embedding.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_text_embedding.cpp",
            "old_class": "TextEmbeddingModule",
            "new_class": "GeneratedTextEmbeddingModule",
            "old_header": "md_text_embedding.hpp",
        },
        "vision_encoder": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_vision_encoder.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_vision_encoder.cpp",
            "old_class": "VisionEncoderModule",
            "new_class": "GeneratedVisionEncoderModule",
            "old_header": "md_vision_encoder.hpp",
        },
        "embedding_merger": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_embedding_merger.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_embedding_merger.cpp",
            "old_class": "EmbeddingMergerModule",
            "new_class": "GeneratedEmbeddingMergerModule",
            "old_header": "md_embedding_merger.hpp",
        },
        "llm_inference": {
            "src_hpp": repo_root / "src/cpp/src/module_genai/modules/md_llm_inference.hpp",
            "src_cpp": repo_root / "src/cpp/src/module_genai/modules/md_llm_inference.cpp",
            "old_class": "LLMInferenceModule",
            "new_class": "GeneratedLlmInferenceModule",
            "old_header": "md_llm_inference.hpp",
        },
    }
    cfg = mapping.get(module_name)
    if not cfg:
        return None
    if not cfg["src_hpp"].exists() or not cfg["src_cpp"].exists():
        return None

    hpp_text = cfg["src_hpp"].read_text(encoding="utf-8", errors="ignore")
    cpp_text = cfg["src_cpp"].read_text(encoding="utf-8", errors="ignore")
    hpp_text = _replace_class_symbols(hpp_text, cfg["old_class"], cfg["new_class"])
    cpp_text = _replace_class_symbols(cpp_text, cfg["old_class"], cfg["new_class"])
    cpp_text = _replace_include_header(cpp_text, cfg["old_header"], generated_header)
    banner = "// Auto-generated parity template from existing module implementation\n"
    return banner + hpp_text, banner + cpp_text


def resolve_effective_spec(skill_spec: Dict[str, object], tools_spec: Dict[str, object]) -> Dict[str, object]:
    base = {
        "spec_name": "qwen2_5_vl_codegen_spec",
        "spec_version": "builtin-1.0",
        "model_type": "qwen2_5_vl",
        "stage_split": {
            "xml_gpu_stages": XML_GPU_STAGES,
            "local_logic_stages": LOCAL_LOGIC_STAGES,
        },
        "default_devices": DEFAULT_DEVICES,
        "output_dir_default": "tools/module_genai/generated_qwen2_5_vl",
    }
    merged = deep_merge_dict(base, skill_spec)
    merged = deep_merge_dict(merged, tools_spec)
    return merged


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "<!DOCTYPE html>" in text[:3000]:
        raise RuntimeError(
            f"{path} 看起来是 HTML 页面而非 Python 源码。"
            "请改用 raw Python 文件（例如 HuggingFace raw 链接下载后的文件）。"
        )
    return text


def analyze_python_model(py_file: Path) -> ModelAnalysis:
    src = read_text(py_file)

    class_names: List[str] = []
    function_names: List[str] = []

    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except SyntaxError:
        # 对自动生成/裁剪文件做降级关键词分析
        pass

    lowered = src.lower()
    found_signals = {
        "image_features": "get_image_features" in lowered or "pixel_values" in lowered,
        "video_features": "get_video_features" in lowered or "pixel_values_videos" in lowered,
        "rope_index": "get_rope_index" in lowered or "rope_delta" in lowered,
        "embed_merge": "masked_scatter" in lowered or "placeholder mask" in lowered,
        "language_model": "language_model" in lowered or "forconditionalgeneration" in lowered,
        "vision_transformer": "visiontransformer" in lowered or "visionblock" in lowered,
    }

    notes: List[str] = []
    if found_signals["image_features"]:
        notes.append("检测到图像特征路径，可映射 ImagePreprocess/VisionEncoder。")
    if found_signals["rope_index"]:
        notes.append("检测到 3D RoPE/rope delta 逻辑，可映射 VisionEncoder -> LLMInference。")
    if found_signals["embed_merge"]:
        notes.append("检测到多模态 embedding 合并逻辑，可映射 EmbeddingMerger。")
    if found_signals["language_model"]:
        notes.append("检测到语言模型主干，可映射 LLMInference。")

    return ModelAnalysis(
        class_names=sorted(set(class_names)),
        function_names=sorted(set(function_names)),
        found_signals=found_signals,
        notes=notes,
    )


def build_yaml_content(
    model_dir: str,
    devices: Dict[str, str],
    cpu_compare: bool,
    module_type_overrides: Dict[str, str] | None = None,
) -> str:
    # cpu_compare=True 时将全部模块强制 CPU，便于和 Python baseline 对齐
    d = {k: ("CPU" if cpu_compare else v) for k, v in devices.items()}
    t = module_type_overrides or {}

    lines = [
        "# Auto-generated by tools/module_genai/qwen2_5_vl_codegen.py",
        "global_context:",
        "  model_type: \"qwen2_5_vl\"",
        "",
        "pipeline_modules:",
        "  pipeline_params:",
        "    type: \"ParameterModule\"",
        "    outputs:",
        "      - name: \"image\"",
        "        type: \"OVTensor\"",
        "      - name: \"prompt\"",
        "        type: \"String\"",
        "",
        "  image_preprocessor:",
        f"    type: \"{t.get('image_preprocessor', 'ImagePreprocessModule')}\"",
        f"    device: \"{d['image_preprocessor']}\"",
        "    inputs:",
        "      - name: \"image\"",
        "        type: \"OVTensor\"",
        "        source: \"pipeline_params.image\"",
        "    outputs:",
        "      - name: \"raw_data\"",
        "        type: \"OVTensor\"",
        "      - name: \"source_size\"",
        "        type: \"VecInt\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      use_python_reference: \"true\"",
        "",
        "  prompt_encoder:",
        f"    type: \"{t.get('prompt_encoder', 'TextEncoderModule')}\"",
        f"    device: \"{d['prompt_encoder']}\"",
        "    inputs:",
        "      - name: \"prompt\"",
        "        type: \"String\"",
        "        source: \"pipeline_params.prompt\"",
        "      - name: \"encoded_image\"",
        "        type: \"OVTensor\"",
        "        source: \"image_preprocessor.raw_data\"",
        "      - name: \"source_size\"",
        "        type: \"VecInt\"",
        "        source: \"image_preprocessor.source_size\"",
        "    outputs:",
        "      - name: \"input_ids\"",
        "        type: \"OVTensor\"",
        "      - name: \"mask\"",
        "        type: \"OVTensor\"",
        "      - name: \"images_sequence\"",
        "        type: \"VecInt\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      use_python_reference: \"false\"",
        "",
        "  text_embedding:",
        f"    type: \"{t.get('text_embedding', 'TextEmbeddingModule')}\"",
        f"    device: \"{d['text_embedding']}\"",
        "    inputs:",
        "      - name: \"input_ids\"",
        "        type: \"OVTensor\"",
        "        source: \"prompt_encoder.input_ids\"",
        "    outputs:",
        "      - name: \"input_embedding\"",
        "        type: \"OVTensor\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      scale_emb: \"1.0\"",
        "      use_python_reference: \"true\"",
        "",
        "  vision_encoder:",
        f"    type: \"{t.get('vision_encoder', 'VisionEncoderModule')}\"",
        f"    device: \"{d['vision_encoder']}\"",
        "    inputs:",
        "      - name: \"preprocessed_image\"",
        "        type: \"OVTensor\"",
        "        source: \"image_preprocessor.raw_data\"",
        "      - name: \"source_size\"",
        "        type: \"VecInt\"",
        "        source: \"image_preprocessor.source_size\"",
        "      - name: \"images_sequence\"",
        "        type: \"VecInt\"",
        "        source: \"prompt_encoder.images_sequence\"",
        "      - name: \"input_ids\"",
        "        type: \"OVTensor\"",
        "        source: \"prompt_encoder.input_ids\"",
        "    outputs:",
        "      - name: \"image_embedding\"",
        "        type: \"OVTensor\"",
        "      - name: \"video_embedding\"",
        "        type: \"OVTensor\"",
        "      - name: \"position_ids\"",
        "        type: \"OVTensor\"",
        "      - name: \"rope_delta\"",
        "        type: \"Int\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      vision_start_token_id: 151652",
        "      use_python_reference: \"true\"",
        "",
        "  embedding_merger:",
        f"    type: \"{t.get('embedding_merger', 'EmbeddingMergerModule')}\"",
        f"    device: \"{d['embedding_merger']}\"",
        "    inputs:",
        "      - name: \"input_ids\"",
        "        type: \"OVTensor\"",
        "        source: \"prompt_encoder.input_ids\"",
        "      - name: \"input_embedding\"",
        "        type: \"OVTensor\"",
        "        source: \"text_embedding.input_embedding\"",
        "      - name: \"image_embedding\"",
        "        type: \"OVTensor\"",
        "        source: \"vision_encoder.image_embedding\"",
        "      - name: \"video_embedding\"",
        "        type: \"OVTensor\"",
        "        source: \"vision_encoder.video_embedding\"",
        "    outputs:",
        "      - name: \"merged_embedding\"",
        "        type: \"OVTensor\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      use_python_reference: \"false\"",
        "",
        "  llm_inference:",
        f"    type: \"{t.get('llm_inference', 'LLMInferenceModule')}\"",
        f"    device: \"{d['llm_inference']}\"",
        "    inputs:",
        "      - name: \"embeds\"",
        "        type: \"OVTensor\"",
        "        source: \"embedding_merger.merged_embedding\"",
        "      - name: \"position_ids\"",
        "        type: \"OVTensor\"",
        "        source: \"vision_encoder.position_ids\"",
        "      - name: \"rope_delta\"",
        "        type: \"Int\"",
        "        source: \"vision_encoder.rope_delta\"",
        "    outputs:",
        "      - name: \"generated_text\"",
        "        type: \"String\"",
        "    params:",
        f"      model_path: \"{model_dir}\"",
        "      max_new_tokens: \"16\"",
        "      do_sample: \"false\"",
        "      top_p: \"1.0\"",
        "      top_k: \"50\"",
        "      temperature: \"1.0\"",
        "      repetition_penalty: \"1.0\"",
        "      use_python_reference: \"true\"",
        "",
        "  pipeline_result:",
        f"    type: \"{t.get('pipeline_result', 'ResultModule')}\"",
        "    inputs:",
        "      - name: \"generated_text\"",
        "        type: \"String\"",
        "        source: \"llm_inference.generated_text\"",
        "",
    ]
    return "\n".join(lines)


def stage_contracts() -> Dict[str, Dict[str, object]]:
    return {
        "image_preprocessor": {
            "xml_runtime": True,
            "python_reference": True,
            "xml_model_hint": ["openvino_vision_embeddings_model.xml"],
            "python_reference_hint": "Qwen2_5_VLForConditionalGeneration.get_image_features 前处理",
            "module_local_logic": [
                "输入图像/批量图像分发",
                "source_size 记录与输出",
                "数据类型/shape 校验",
            ],
        },
        "prompt_encoder": {
            "xml_runtime": False,
            "python_reference": False,
            "xml_model_hint": [],
            "python_reference_hint": "无（应在 C++ 模块内实现）",
            "module_local_logic": [
                "chat template 应用",
                "视觉 placeholder 展开（image/video pad）",
                "tokenizer encode 输出 input_ids/mask/images_sequence",
            ],
        },
        "text_embedding": {
            "xml_runtime": True,
            "python_reference": True,
            "xml_model_hint": ["openvino_text_embeddings_model.xml 或等价 embeddings IR"],
            "python_reference_hint": "language_model.get_input_embeddings 对齐",
            "module_local_logic": [
                "input_ids shape 校验",
                "scale_emb 处理",
                "remote/local tensor 策略",
            ],
        },
        "vision_encoder": {
            "xml_runtime": True,
            "python_reference": True,
            "xml_model_hint": ["openvino_vision_embeddings_merger_model.xml"],
            "python_reference_hint": "visual forward / get_window_index / rot_pos_emb 对齐",
            "module_local_logic": [
                "window_index 与 rotary_pos_emb 计算",
                "position_ids 与 rope_delta 计算",
                "image/video embedding 拆分",
            ],
        },
        "embedding_merger": {
            "xml_runtime": False,
            "python_reference": False,
            "xml_model_hint": [],
            "python_reference_hint": "无（应在 C++ 模块内实现）",
            "module_local_logic": [
                "根据 image_pad/video_pad token 替换文本 embedding",
                "placeholder 数量与特征数量一致性校验",
            ],
        },
        "llm_inference": {
            "xml_runtime": True,
            "python_reference": True,
            "xml_model_hint": ["openvino_language_model.xml / KV cache 相关 IR"],
            "python_reference_hint": "Qwen2_5_VLForConditionalGeneration.generate",
            "module_local_logic": [
                "generation_config 覆盖",
                "position_ids + rope_delta 对齐传递",
                "批量请求包装/解码",
            ],
        },
    }


def _camel_case(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def generate_module_skeleton(module_name: str, device: str, contracts_map: Dict[str, Dict[str, object]]) -> Tuple[str, str, str]:
    cls = f"Generated{_camel_case(module_name)}Module"
    file_base = f"md_generated_{module_name}"

    contracts = contracts_map.get(module_name, {})
    local_logic = contracts.get("module_local_logic", [])
    local_logic_comment = "\n".join(f"    // - {item}" for item in local_logic)

    hpp = f'''// Auto-generated skeleton
#pragma once

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"

namespace ov {{
namespace genai {{
namespace module {{

class {cls} : public IBaseModule {{
    DeclareModuleConstructor({cls});

private:
    bool m_use_python_reference = false;
    bool initialize();
    void run_with_python_reference();
}};

REGISTER_MODULE_CONFIG({cls});

}}  // namespace module
}}  // namespace genai
}}  // namespace ov
'''

    cpp = f'''// Auto-generated skeleton
#include "{file_base}.hpp"

#include "module_genai/module_factory.hpp"
#include <cstdlib>

namespace ov {{
namespace genai {{
namespace module {{

GENAI_REGISTER_MODULE_SAME({cls});

void {cls}::print_static_config() {{
    std::cout << R"(
  {module_name}:
    type: "{cls}"
    device: "{device}"
    # TODO: declare inputs/outputs/params
    )" << std::endl;
}}

{cls}::{cls}(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {{
    if (!initialize()) {{
        GENAI_ERR("Failed to initialize {cls}");
    }}
}}

{cls}::~{cls}() {{}}

bool {cls}::initialize() {{
    auto it_ref = module_desc->params.find("use_python_reference");
    if (it_ref != module_desc->params.end()) {{
        const auto& v = it_ref->second;
        m_use_python_reference = (v == "true" || v == "1" || v == "True");
    }}
    return true;
}}

void {cls}::run_with_python_reference() {{
    // GPU阶段建议：通过 Python 原始实现做对齐。
    // TODO: 使用 pybind11 或 subprocess + 文件交换实现 I/O 映射。
    // 示例：std::system("python3 qwen2_5_vl_reference_bridge.py --stage {module_name}");
}}

void {cls}::run() {{
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    // 约定：GPU+use_python_reference=true 时，先走 Python baseline。
    if (module_desc->device == "GPU" && m_use_python_reference) {{
        run_with_python_reference();
        return;
    }}

    // TODO: 在此填充模块内本地逻辑实现（与现有 md_* 模块风格一致）
{local_logic_comment}
}}

}}  // namespace module
}}  // namespace genai
}}  // namespace ov
'''

    return file_base, hpp, cpp


def build_reference_bridge_script() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_image(path: str):
    return Image.open(path).convert("RGB")


def run_full_generate(model_id_or_path: str, prompt: str, image_path: str, max_new_tokens: int = 16):
    device = torch.device("cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id_or_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": load_image(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[load_image(image_path)], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="llm_full", choices=["llm_full"])
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = {
        "stage": args.stage,
        "generated_text": run_full_generate(args.model, args.prompt, args.image, args.max_new_tokens),
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


def build_compare_script() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from openvino import Tensor
import openvino_genai


def load_image_as_ov_tensor(image_path: str) -> Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return Tensor(arr)


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--python-bridge", required=True)
    parser.add_argument("--hf-model", required=True, help="Qwen2.5-VL HF model id or local path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    # 1) ModulePipeline(CPU YAML) 输出
    pipe = openvino_genai.ModulePipeline(args.yaml)
    pipe.generate(image=load_image_as_ov_tensor(args.image), prompt=args.prompt)
    ov_text = str(pipe.get_output("generated_text"))

    # 2) Python 原始模型输出
    ref_json = Path(args.report).with_suffix(".ref.json")
    cmd = [
        sys.executable,
        args.python_bridge,
        "--stage", "llm_full",
        "--model", args.hf_model,
        "--prompt", args.prompt,
        "--image", args.image,
        "--max-new-tokens", str(args.max_new_tokens),
        "--out", str(ref_json),
    ]
    subprocess.run(cmd, check=True)
    ref = json.loads(ref_json.read_text(encoding="utf-8"))
    py_text = ref["generated_text"]

    # 3) 粗粒度一致性（可替换成 BLEU/ROUGE）
    ov_n = norm_text(ov_text)
    py_n = norm_text(py_text)
    exact = ov_n == py_n
    contain = (ov_n in py_n) or (py_n in ov_n)

    report = {
        "ov_text": ov_text,
        "py_text": py_text,
        "exact_match": exact,
        "contains_match": contain,
        "pass": exact or contain,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def write_file(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def emit_local_stage_parity_references(repo_root: Path, out_dir: Path) -> Dict[str, str]:
    parity_dir = out_dir / "modules" / "parity_existing"
    mapping = {
        "prompt_encoder_hpp": repo_root / "src/cpp/src/module_genai/modules/md_text_encoder.hpp",
        "prompt_encoder_cpp": repo_root / "src/cpp/src/module_genai/modules/md_text_encoder.cpp",
        "embedding_merger_hpp": repo_root / "src/cpp/src/module_genai/modules/md_embedding_merger.hpp",
        "embedding_merger_cpp": repo_root / "src/cpp/src/module_genai/modules/md_embedding_merger.cpp",
    }
    output_files: Dict[str, str] = {}
    for key, src in mapping.items():
        if src.exists():
            dst = parity_dir / src.name
            copy_file(src, dst)
            output_files[key] = str(dst)
    return output_files


def emit_sync_generated_modules_script(out_dir: Path) -> str:
    script_path = out_dir / "sync_generated_modules_to_src.sh"
    content = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=\"$(cd \"$(dirname \"$0\")/../../..\" && pwd)\"
GEN_DIR=\"$ROOT_DIR/tools/module_genai/generated_qwen2_5_vl/modules\"
DST_DIR=\"$ROOT_DIR/src/cpp/src/module_genai/modules/generated\"

mkdir -p \"$DST_DIR\"
cp \"$GEN_DIR\"/md_generated_image_preprocessor.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_image_preprocessor.cpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_prompt_encoder.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_prompt_encoder.cpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_text_embedding.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_text_embedding.cpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_vision_encoder.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_vision_encoder.cpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_embedding_merger.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_embedding_merger.cpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_llm_inference.hpp \"$DST_DIR\"/
cp \"$GEN_DIR\"/md_generated_llm_inference.cpp \"$DST_DIR\"/

echo \"Synced generated modules to: $DST_DIR\"
echo \"Now rebuild project. src/cpp/CMakeLists.txt uses recursive glob under src/, so files are picked automatically.\"
"""
    write_file(script_path, content, executable=True)
    return str(script_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen2.5-VL module YAML + skeletons + compare scripts")
    parser.add_argument("--python-model-file", required=True, help="Path to modeling_qwen2_5_vl.py")
    parser.add_argument("--model-dir", required=True, help="OpenVINO model directory used in YAML")
    parser.add_argument("--output-dir", required=True, help="Directory to place generated artifacts")
    parser.add_argument("--skill-md", default="", help="Skill markdown path with CODEGEN_SPEC_JSON")
    parser.add_argument("--tools-md", default="", help="Tools markdown path with CODEGEN_SPEC_JSON")
    parser.add_argument("--gpu-policy", choices=["hybrid", "all-gpu", "all-cpu"], default="hybrid")
    parser.add_argument("--emit-cpu-compare-yaml", action="store_true", help="Also emit all-CPU YAML for baseline test")
    parser.add_argument("--emit-generated-types-yaml", action="store_true", help="Emit YAML variant that uses GeneratedPromptEncoderModule/GeneratedEmbeddingMergerModule")
    parser.add_argument("--emit-iteration-plan", action="store_true", help="Emit md-driven iteration checklist into output dir")
    parser.add_argument("--emit-local-parity-reference", action="store_true", help="Copy existing md_text_encoder/md_embedding_merger into output as parity references")
    parser.add_argument("--disable-local-parity-transform", action="store_true", help="Disable class-renamed parity generation for local stages and use skeletons")
    args = parser.parse_args()

    py_file = Path(args.python_model_file).resolve()
    out_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]

    skill_md_path = Path(args.skill_md).resolve() if args.skill_md else None
    tools_md_path = Path(args.tools_md).resolve() if args.tools_md else None
    skill_spec = extract_codegen_spec_from_md(skill_md_path)
    tools_spec = extract_codegen_spec_from_md(tools_md_path)
    effective_spec = resolve_effective_spec(skill_spec, tools_spec)

    analysis = analyze_python_model(py_file)

    spec_devices = effective_spec.get("default_devices", DEFAULT_DEVICES)
    if isinstance(spec_devices, dict):
        devices = dict(spec_devices)
    else:
        devices = dict(DEFAULT_DEVICES)
    if args.gpu_policy == "all-gpu":
        for k in devices:
            devices[k] = "GPU"
    elif args.gpu_policy == "all-cpu":
        for k in devices:
            devices[k] = "CPU"

    # YAMLs
    yaml_gpu = build_yaml_content(args.model_dir, devices, cpu_compare=False)
    yaml_gpu_path = out_dir / "config.generated.gpu.yaml"
    write_file(yaml_gpu_path, yaml_gpu)

    if args.emit_cpu_compare_yaml:
        yaml_cpu = build_yaml_content(args.model_dir, devices, cpu_compare=True)
        yaml_cpu_path = out_dir / "config.generated.cpu_compare.yaml"
        write_file(yaml_cpu_path, yaml_cpu)

    yaml_generated_types_path = None
    if args.emit_generated_types_yaml:
        type_overrides = {
            "image_preprocessor": "GeneratedImagePreprocessorModule",
            "prompt_encoder": "GeneratedPromptEncoderModule",
            "text_embedding": "GeneratedTextEmbeddingModule",
            "vision_encoder": "GeneratedVisionEncoderModule",
            "embedding_merger": "GeneratedEmbeddingMergerModule",
            "llm_inference": "GeneratedLlmInferenceModule",
        }
        yaml_generated_types = build_yaml_content(
            args.model_dir,
            devices,
            cpu_compare=False,
            module_type_overrides=type_overrides,
        )
        yaml_generated_types_path = out_dir / "config.generated.generated_types.yaml"
        write_file(yaml_generated_types_path, yaml_generated_types)

    # Skeleton modules
    modules_dir = out_dir / "modules"
    contracts = stage_contracts()
    parity_transform_applied: List[str] = []
    for stage in [
        "image_preprocessor",
        "prompt_encoder",
        "text_embedding",
        "vision_encoder",
        "embedding_merger",
        "llm_inference",
    ]:
        base, hpp, cpp = generate_module_skeleton(stage, devices[stage], contracts)

        # 对所有已映射阶段，优先从现有实现自动生成“可编译模板”（类名替换版）
        if not args.disable_local_parity_transform:
            transformed = generate_stage_from_existing(stage, repo_root, f"{base}.hpp")
            if transformed is not None:
                hpp, cpp = transformed
                parity_transform_applied.append(stage)

        write_file(modules_dir / f"{base}.hpp", hpp)
        write_file(modules_dir / f"{base}.cpp", cpp)

    # Python bridge + compare
    py_ref_dir = out_dir / "python_ref"
    write_file(py_ref_dir / "qwen2_5_vl_reference_bridge.py", build_reference_bridge_script(), executable=True)
    write_file(py_ref_dir / "compare_pipeline_with_reference.py", build_compare_script(), executable=True)

    parity_refs: Dict[str, str] = {}
    if args.emit_local_parity_reference:
        parity_refs = emit_local_stage_parity_references(repo_root, out_dir)

    sync_script = emit_sync_generated_modules_script(out_dir)

    # Analysis report
    stage_split = effective_spec.get("stage_split", {}) if isinstance(effective_spec, dict) else {}
    xml_gpu_stages = stage_split.get("xml_gpu_stages", XML_GPU_STAGES) if isinstance(stage_split, dict) else XML_GPU_STAGES
    local_logic_stages = stage_split.get("local_logic_stages", LOCAL_LOGIC_STAGES) if isinstance(stage_split, dict) else LOCAL_LOGIC_STAGES

    md_sources = {
        "skill_md": str(skill_md_path) if skill_md_path else None,
        "tools_md": str(tools_md_path) if tools_md_path else None,
        "skill_spec_loaded": bool(skill_spec),
        "tools_spec_loaded": bool(tools_spec),
    }

    report = {
        "python_model_file": str(py_file),
        "md_sources": md_sources,
        "effective_spec": effective_spec,
        "classes_count": len(analysis.class_names),
        "functions_count": len(analysis.function_names),
        "signals": analysis.found_signals,
        "notes": analysis.notes,
        "stage_split": {
            "xml_gpu_stages": xml_gpu_stages,
            "local_logic_stages": local_logic_stages,
            "contracts": contracts,
        },
        "generated": {
            "yaml_gpu": str(yaml_gpu_path),
            "yaml_cpu_compare": str(out_dir / "config.generated.cpu_compare.yaml") if args.emit_cpu_compare_yaml else None,
            "yaml_generated_types": str(yaml_generated_types_path) if yaml_generated_types_path else None,
            "modules_dir": str(modules_dir),
            "python_ref_dir": str(py_ref_dir),
            "local_stage_parity_references": parity_refs,
            "parity_transform_applied_stages": parity_transform_applied,
            "sync_generated_modules_script": sync_script,
        },
    }
    write_file(out_dir / "analysis_report.json", json.dumps(report, ensure_ascii=False, indent=2))

    if args.emit_iteration_plan:
        iteration_lines = [
            "# Auto Iteration Checklist",
            "",
            "1. 更新 skill/tools md 中 CODEGEN_SPEC_JSON",
            "2. 运行 codegen 并检查 analysis_report.effective_spec",
            "3. 对比生成 YAML 与现有模块端口/参数一致性",
            "4. 打开 use_python_reference，先打通 GPU XML 阶段",
            "5. 逐阶段关闭 use_python_reference，迁移到本地模块实现",
        ]
        write_file(out_dir / "ITERATION_CHECKLIST.md", "\n".join(iteration_lines))

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
