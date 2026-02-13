#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text/image/mixed consistency validation")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--python-bridge", required=True)
    parser.add_argument("--compare-script", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--text-image", required=True, help="Image path for text-dominant case")
    parser.add_argument("--image-image", required=True, help="Image path for image-dominant case")
    parser.add_argument("--mixed-image", required=True, help="Image path for mixed case")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "text",
            "prompt": "Repeat exactly this sentence: openvino module pipeline validation.",
            "image": args.text_image,
        },
        {
            "name": "image",
            "prompt": "Describe the main object in this image using one short sentence.",
            "image": args.image_image,
        },
        {
            "name": "mixed",
            "prompt": "Based on the image, answer this question in one word: what animal is shown?",
            "image": args.mixed_image,
        },
    ]

    results = []
    for c in cases:
        out_report = report_dir / f"{c['name']}.report.json"
        cmd = [
            sys.executable,
            args.compare_script,
            "--yaml",
            args.yaml,
            "--python-bridge",
            args.python_bridge,
            "--hf-model",
            args.hf_model,
            "--prompt",
            c["prompt"],
            "--image",
            c["image"],
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--report",
            str(out_report),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        entry = {
            "case": c["name"],
            "prompt": c["prompt"],
            "image": c["image"],
            "report_file": str(out_report),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if out_report.exists():
            try:
                entry["compare"] = json.loads(out_report.read_text(encoding="utf-8"))
            except Exception:
                entry["compare"] = None
        results.append(entry)

    summary = {
        "all_pass": all(r.get("compare", {}).get("pass", False) for r in results if r.get("returncode") == 0)
        and all(r.get("returncode") == 0 for r in results),
        "results": results,
    }

    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
