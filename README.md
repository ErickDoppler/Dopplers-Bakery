# Dopplers Bakery

Compact standalone baker for text (LLM), speech (TTS), images (Krea2) and audio (Stable Audio 3).
One EXE, no Python, no ComfyUI. Window UI when started plain, command line when given arguments.

- Models: `models\` next to the exe (first launch offers to download them). ComfyUI model folders can be added from Models > Add models folder.
- Output: `output\<tab>\` next to the exe.
- Build: `build.bat` (toolchain in c:workenv). The inference runtime is a set of prebuilt DLLs in `runtime` (llama.cpp release b12282); see docs/KREA2-PORT.md. Result: `dist`.
- CLI: `DopplersBakery.exe llm --prompt "hello"`, `DopplersBakery.exe models list`, `DopplersBakery.exe image --help`.

## Engine status
| Tab | Engine | State |
|---|---|---|
| LLM | llama.cpp / ggml, CPU | implemented |
| Text to image | Krea2 (Qwen3-VL encoder + DiT + Wan VAE) on ggml, CUDA/Vulkan/CPU | implemented |
| TTS | Qwen3-TTS 12Hz (talker + code predictor + speech decoder) on ggml, CUDA/Vulkan/CPU | implemented (CustomVoice, VoiceDesign) |
| Audio | Stable Audio 3 medium (T5Gemma + DiT + audio VAE) on ggml | port pending |

Ports follow ComfyUI's Python graphs (`comfy/ldm`, `comfy/text_encoders`, `comfy_extras`, `qwen3-tts-comfyui`).
