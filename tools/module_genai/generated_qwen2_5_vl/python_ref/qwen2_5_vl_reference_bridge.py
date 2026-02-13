#!/usr/bin/env python3
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
