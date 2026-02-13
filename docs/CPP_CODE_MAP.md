# C++ Code Map (src/cpp/src)

This document maps major directories, shared modules and where they are referenced.

## Top-level shared modules

- Logger
  - Files: `src/cpp/src/logger.hpp`, `src/cpp/src/logger.cpp`
  - Used in: whisper, rag, sampling, structured output and many other files

- SynchronizedQueue
  - Files: `src/cpp/src/synchronized_queue.hpp`
  - Used in: generation stream, continuous batching threaded streamer, image threaded callbacks

- CircularBufferQueue
  - Files: `src/cpp/src/circular_buffer_queue.hpp`
  - Used in: tokenizer implementation to manage infer request pools

- Tokenizer
  - Files: `src/cpp/src/tokenizer/tokenizer_impl.hpp`, `src/cpp/src/tokenizer/tokenizer_impl.cpp`, `tokenizers_path`, `make_tokenizer_stateful`
  - Used in: generation_config, sampling/sampler, structured output controller, continuous batching pipelines

- LoRA
  - Files: `src/cpp/src/lora/helper.hpp/.cpp`, `lora/common.hpp`, `lora/names_mapping.*`, `lora/adapter.cpp`, `lora/safetensors.c`
  - Used in: continuous batching implementation, llm pipelines, image generation models

- Continuous batching
  - Files: `src/cpp/src/continuous_batching/*` (pipeline_impl, pipeline_base, cache_manager, cache_eviction, scheduler, model_runner, etc.)
  - Used in: speculative decoding, prompt lookup, visual language pipelines, and llm adapters

- Sampling & Logit processing
  - Files: `src/cpp/src/sampling/*`
  - Used in: llm pipelines, whisper, structured output

- GGUF utilities
  - Files: `src/cpp/src/gguf_utils/*`
  - Used in: tokenizer building and gguf model handling

## Directory snapshot (major files only)

- `src/cpp/src/`
  - `logger.hpp`, `logger.cpp`
  - `utils.hpp`, `utils.cpp`
  - `json_utils.hpp`, `json_container.cpp`
  - Tokenizer: `tokenizer/tokenizer_impl.hpp`, `tokenizer/tokenizer_impl.cpp`, `tokenizer/*.hpp/.cpp`
  - LORA: `lora/helper.hpp`, `lora/helper.cpp`, `lora/common.hpp`, `lora/names_mapping.*`, `lora/adapter.cpp`
  - LLM Pipelines: `llm/pipeline_base.hpp`, `llm/pipeline.cpp`, `llm/pipeline_stateful.*`, `llm/pipeline_static.*`
  - Continuous batching: `continuous_batching/pipeline_impl.hpp`, `continuous_batching/pipeline_impl.cpp`, `continuous_batching/*`
  - Sampling: `sampling/sampler.hpp`, `sampling/sampler.cpp`, `sampling/logit_processor.hpp`
  - Image generation: `image_generation/*` (pipelines, models, schedulers)
  - Whisper: `whisper/*` (pipeline, models, features)
  - Visual Language: `visual_language/*` (pipeline, embedding, encoders, model classes)
  - GGUF: `gguf_utils/*`
  - Generation helpers: `generation_handle.cpp`, `generation_stream.hpp`, `text_streamer.cpp`

## How to use these artifacts

- `tools/analysis/shared_modules.csv` contains a CSV mapping: module, definition files, example references.
- `docs/CPP_CODE_MAP.md` (this file) provides a quick human-readable map of shared modules and where to find them.

If you want more detail (full file-by-file include matrix or a DOT file for dependency visualization), tell me which format you prefer and I will generate it.
