#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
GEN_DIR="$ROOT_DIR/tools/module_genai/generated_qwen2_5_vl/modules"
DST_DIR="$ROOT_DIR/src/cpp/src/module_genai/modules/generated"

mkdir -p "$DST_DIR"
cp "$GEN_DIR"/md_generated_image_preprocessor.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_image_preprocessor.cpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_prompt_encoder.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_prompt_encoder.cpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_text_embedding.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_text_embedding.cpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_vision_encoder.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_vision_encoder.cpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_embedding_merger.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_embedding_merger.cpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_llm_inference.hpp "$DST_DIR"/
cp "$GEN_DIR"/md_generated_llm_inference.cpp "$DST_DIR"/

echo "Synced generated modules to: $DST_DIR"
echo "Now rebuild project. src/cpp/CMakeLists.txt uses recursive glob under src/, so files are picked automatically."
