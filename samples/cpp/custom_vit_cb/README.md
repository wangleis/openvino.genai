# C++ custom ViT + continuous batching (inputs_embeds)

This draft sample shows how to pass user-produced `inputs_embeds` directly into `ov::genai::ContinuousBatchingPipeline` for inference. The ViT (or any custom visual encoder) is expected to be implemented by the user; the sample only demonstrates how to feed its output into continuous batching.

## Build

Use the provided build script:

`./build.sh`

> If OpenVINO GenAI is not found automatically, set `OpenVINOGenAI_DIR` to the folder that contains `OpenVINOGenAIConfig.cmake` before running the script.

## Run

`./build/custom_vit_cb <MODEL_DIR>`

`MODEL_DIR` must point to a model export that supports `inputs_embeds` in the continuous batching pipeline. Replace the placeholder embedding creation in [custom_vit_cb.cpp](custom_vit_cb.cpp) with your custom ViT inference code.
