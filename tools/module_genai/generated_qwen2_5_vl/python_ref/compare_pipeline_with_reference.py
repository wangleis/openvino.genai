#!/usr/bin/env python3
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
