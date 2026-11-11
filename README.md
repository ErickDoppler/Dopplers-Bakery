# Dopplers Bakery

An offline studio for **chat, images, music, sound effects and speech** on your own Windows PC.
A native C++ app (ggml / llama.cpp), no Python, no ComfyUI, no cloud. Talk to the assistant and it makes pictures, audio and a spoken reply; edit your own pictures; use the same chat from a browser.

> ## ⚠️ Read this first: the app downloads gigabytes of data
> The program itself is small (~70 MB), but it **does not contain any AI models**. To work it needs about **36 GB** of model files
> (all features ≈ 36 GB; images only ≈ 18 GB; speech only ≈ 4.5 GB). On first launch it offers to download them from Hugging Face
> (resumable). Make sure you have **free disk space, an unmetered connection and time**. A GPU with 8 GB+ VRAM (Vulkan) is recommended;
> without a GPU it runs on the CPU, much more slowly.

## Features
- **Chat** (first tab): ChatGPT-style. Ask for *"a lighthouse at sunset"*, *"20 seconds of calm piano"*, *"a door creaking"*; the assistant calls the image / audio / speech tools and answers (optionally out loud).
- **Pictures in chat**: attach with the **+** button, drag & drop, or Ctrl+V; then say how to change it (image-to-image).
- **Chat in the browser**: the same chat on `http://127.0.0.1:8787` (menu *Web*), images and sound play in the page. *Share on the local network* is opt-in and protected by a secret key in the link.
- **Tabs**: LLM (Qwen), TTS (Qwen3-TTS, incl. voice cloning with the Base model), Text to image (Krea 2, image-to-image), Audio (Stable Audio 3, MP3/WAV).
- Job **queue** (pause, cancel, reorder; hide or resize the panel), command line mode, models are plain files you can reuse from ComfyUI folders.

## Requirements
- Windows 10/11 64-bit, CPU with AVX2 (2013+), ~36 GB free disk space, 16 GB+ RAM recommended.
- GPU with current Vulkan drivers (NVIDIA / AMD / Intel) for usable speed. Measured on an RTX 3080 Ti: about 1.5 s per image step at 720×720; 20 s of audio in about 8 s (after the models are loaded).

## Install
1. Download `DopplersBakery-<version>-win64.zip` from the **Releases** page and unpack it anywhere (keep the folder together).
2. Run `DopplersBakery.exe`. A dialog offers to download the models below, or use **Search in folder** if you already have them.
3. Models go to `models\`, results to `output\` (both next to the exe).

## Files the app needs (exact types and current download links)
The app reads **three kinds of model files**: `.gguf` (language model), `.safetensors` (diffusion / audio / text encoder / VAE) and **one model folder** for speech (`.safetensors` plus `.json` and tokenizer files). No other formats (no `.ckpt`, `.pt`, `.bin`).

| Feature (tab) | Role | File | Type | Size | Where it goes | Download |
|---|---|---|---|---|---|---|
| Chat / LLM | language model | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `.gguf` | 2.5 GB | `models\LLM\` | [link](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) |
| Speech (TTS) | model folder | `Qwen3-TTS-12Hz-1.7B-CustomVoice` (whole repo, incl. `speech_tokenizer\`) | folder: `model.safetensors` 3.8 GB, `speech_tokenizer\model.safetensors` 0.7 GB, `config.json`, `generation_config.json`, `merges.txt`, `vocab.json`, `tokenizer_config.json`, `preprocessor_config.json` | 4.5 GB | `models\qwen-tts\Qwen3-TTS-12Hz-1.7B-CustomVoice\` | [repo](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |
| Images | diffusion model | `krea2_turbo_fp8.safetensors` | `.safetensors` | 12.9 GB | `models\diffusion_models\` | [link](https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/krea2_turbo_fp8.safetensors) |
| Images | text encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `.safetensors` | 5.2 GB | `models\text_encoders\` | [link](https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors) |
| Images | VAE | `qwen_image_vae.safetensors` | `.safetensors` | 0.25 GB | `models\vae\` | [link](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) |
| Audio | audio model | `stable_audio_3_medium.safetensors` | `.safetensors` | 9.2 GB | `models\checkpoints\` | [link](https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_medium.safetensors) |
| Audio | text encoder | `t5gemma_b_b_ul2.safetensors` | `.safetensors` | 1.2 GB | `models\text_encoders\` | [link](https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main/text_encoders/t5gemma_b_b_ul2.safetensors) |
| **Total** | | | | **≈ 35.8 GB** | | |

Optional:
- Voice cloning in the TTS tab: [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (another model folder, similar size) in `models\qwen-tts\`.
- LoRA for images (`.safetensors`) and for the LLM (`.gguf`); the audio prompt enhancer uses the LLM tab's `.gguf`.
- Other builds of the same models work if they have the same tensor layout; use **Models > Add models folder** to point at an existing ComfyUI `models` folder.

Notes on the sources: these are vendor / community mirrors on Hugging Face and the links may change (checked 2026-11-11). The Krea 2 files come from a community FP8 repack
([AlperKTS/Krea2_FP8](https://huggingface.co/AlperKTS/Krea2_FP8)) of the official [Krea 2](https://huggingface.co/krea/Krea-2-Raw) model, whose own repository is gated and under the Krea 2 license.
Check the license of each model before using or sharing its outputs.

## Command line
```
DopplersBakery.exe chat  --say "make a picture of a fox" [--image file.png] [--voice]
DopplersBakery.exe serve [--port 8787] [--lan]        chat in a browser, no window
DopplersBakery.exe image --prompt "..." [--init_image file.png --denoise 0.6]
DopplersBakery.exe audio --prompt "calm piano" --length 20
DopplersBakery.exe tts   --text "Hello" --speaker Ryan
DopplersBakery.exe llm   --prompt "hello"
DopplersBakery.exe models list | scan | download <id>
```
`<tab> --help` lists every setting. The environment variable `BAKERY_BACKEND=cpu|vulkan|cuda` forces a backend.

## Build from source
Needs the toolchain in `c:\workenv` (w64devkit, cmake, ninja) and the prebuilt llama.cpp release **b12282** DLLs in `runtime\` (the same commit as the headers in `third_party\llama.cpp`).
`build.bat` builds into `dist\`; `package.bat` assembles `release\DopplersBakery\` (Vulkan + CPU runtime, no CUDA). More in `docs\`.

## How it works
Re-implementations of the ComfyUI graphs on ggml, validated against reference tensors: Krea 2 (Qwen3-VL text tower → DiT → Wan VAE, with the VAE encoder for image-to-image), Qwen3-TTS 12 Hz (talker + code predictor + speech decoder, ECAPA/Mimi encoders for cloning), Stable Audio 3 (T5Gemma → differential-attention DiT → audio VAE). See `docs\KREA2-PORT.md`, `docs\TTS-PORT.md`, `docs\AUDIO-PORT.md`.

## License
Free for **personal use**; **enterprise / organizational use requires a paid license of US$100 per workstation** that launches the app. Source-available, not open source. See [LICENSE](LICENSE) for the exact terms (contact: the repository owner on GitHub). The AI models are not covered by this license and have their own terms; llama.cpp / ggml is MIT-licensed.
