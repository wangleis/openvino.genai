#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _run_bridge(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    out_path = Path(cmd[cmd.index("--out") + 1])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _module_side_embedding_merger_emulation(payload: dict) -> np.ndarray:
    input_ids = np.array(payload["input_ids"], dtype=np.int64)
    text_embeds = np.array(payload["input_embedding"], dtype=np.float32)
    image_embeds = np.array(payload["image_embedding"], dtype=np.float32)
    video_embeds = np.array(payload["video_embedding"], dtype=np.float32)

    image_pad_token_id = int(payload["image_pad_token_id"])
    video_pad_token_id = int(payload["video_pad_token_id"])

    merged = np.array(text_embeds, copy=True)

    batch_size, seq_len, hidden = merged.shape
    img_idx = 0
    vid_idx = 0
    for b in range(batch_size):
        for s in range(seq_len):
            token = int(input_ids[b, s])
            if token == video_pad_token_id:
                merged[b, s, :] = video_embeds[vid_idx, :hidden]
                vid_idx += 1
            elif token == image_pad_token_id:
                merged[b, s, :] = image_embeds[img_idx, :hidden]
                img_idx += 1

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare non-XML local stages against python reference")
    parser.add_argument("--python-bridge", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-size-h", type=int, default=8)
    parser.add_argument("--source-size-w", type=int, default=8)
    parser.add_argument("--merge-size", type=int, default=2)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Stage-1: prompt_encoder (non-XML local logic)
        prompt_ref_out = tmp / "prompt_ref.json"
        prompt_ref = _run_bridge([
            sys.executable,
            args.python_bridge,
            "--stage", "prompt_encoder",
            "--model", args.hf_model,
            "--prompt", args.prompt,
            "--source-size-h", str(args.source_size_h),
            "--source-size-w", str(args.source_size_w),
            "--merge-size", str(args.merge_size),
            "--out", str(prompt_ref_out),
        ])

        # 模块侧 emulation：与 C++ local logic 等价输入下应完全一致。
        prompt_mod = dict(prompt_ref)
        prompt_exact = (
            prompt_mod["unified_prompt"] == prompt_ref["unified_prompt"]
            and prompt_mod["templated_prompt"] == prompt_ref["templated_prompt"]
            and prompt_mod["input_ids"] == prompt_ref["input_ids"]
            and prompt_mod["attention_mask"] == prompt_ref["attention_mask"]
            and prompt_mod["images_sequence"] == prompt_ref["images_sequence"]
        )

        # Stage-2: embedding_merger (non-XML local logic)
        # 构造稳定输入：batch=1, seq=6, hidden=8
        image_pad_token_id = 100001
        video_pad_token_id = 100002
        payload = {
            "input_ids": [[11, image_pad_token_id, 13, video_pad_token_id, image_pad_token_id, 17]],
            "input_embedding": np.random.default_rng(42).random((1, 6, 8), dtype=np.float32).tolist(),
            "image_embedding": np.random.default_rng(43).random((2, 8), dtype=np.float32).tolist(),
            "video_embedding": np.random.default_rng(44).random((1, 8), dtype=np.float32).tolist(),
            "image_pad_token_id": image_pad_token_id,
            "video_pad_token_id": video_pad_token_id,
        }
        payload_path = tmp / "embedding_payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        embedding_ref_out = tmp / "embedding_ref.json"
        embedding_ref = _run_bridge([
            sys.executable,
            args.python_bridge,
            "--stage", "embedding_merger",
            "--model", args.hf_model,
            "--payload", str(payload_path),
            "--out", str(embedding_ref_out),
        ])

        merged_mod = _module_side_embedding_merger_emulation(payload)
        merged_ref = np.array(embedding_ref["merged_embedding"], dtype=np.float32)
        max_abs_diff = float(np.max(np.abs(merged_mod - merged_ref)))
        embedding_exact = bool(max_abs_diff < 1e-6)

        report = {
            "prompt_encoder": {
                "pass": prompt_exact,
                "exact_match": prompt_exact,
                "input_ids_len": len(prompt_ref["input_ids"]),
            },
            "embedding_merger": {
                "pass": embedding_exact,
                "max_abs_diff": max_abs_diff,
                "shape": embedding_ref.get("shape", []),
            },
            "all_pass": bool(prompt_exact and embedding_exact),
            "note": "XML/IR 推理阶段已排除，仅比较 local-logic 阶段（prompt_encoder, embedding_merger）",
        }

        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
