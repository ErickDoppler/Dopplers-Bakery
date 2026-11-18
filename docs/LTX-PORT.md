# LTX-2.3 native port (Text/Image to video tab)

Goal: run `VIDEO_WORKFLOW.json` (LTX-2.3 22B audio+video, image-to-video) inside Dopplers Bakery with C++/ggml only, no ComfyUI, no Python.
Files the user downloads: `ltx-2.3-22b-dev-fp8.safetensors` (27 GB), `gemma_3_12B_it_fp4_mixed.safetensors` (8.8 GB),
`ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (0.9 GB), `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` (2.6 GB),
optional `gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors` (0.6 GB).

## What the checkpoint contains (survey of the safetensors header, 8940 tensors)
| Part | Prefix | Size | Notes |
|---|---|---|---|
| Transformer blocks | `model.diffusion_model.transformer_blocks.N` | 18.8 GB | 48 blocks, video stream width 4096, audio stream 2048, 32 heads |
| Rest of the DiT | `model.diffusion_model.*` | 4.5 GB | patchify_proj (128->4096), audio_patchify_proj (128->2048), proj_out, adaLN-single modules, prompt/AV cross-attn adaLN, 8-block text "connectors" with 128 learnable registers |
| Text projection | `text_embedding_projection.{video,audio}_aggregate_embed` | 2.2 GB | maps the stacked Gemma hidden states to the two streams |
| Video VAE | `vae.encoder` / `vae.decoder` + `per_channel_statistics` | 1.4 GB | needed: encoder for the start image, decoder for the output |
| Audio VAE | `audio_vae.encoder/decoder` | 0.1 GB | decoder only (mel latents -> mel) |
| Vocoder | `vocoder.vocoder` (BigVGAN-style, anti-aliased activations) + `vocoder.bwe_generator` + `mel_stft` | 0.25 GB | mel -> waveform, bandwidth extension |

dtypes: F8_E4M3 (1496 tensors, the big linear weights, per-layer metadata in `_quantization_metadata`), BF16, F32.
Gemma 3 12B file: NVFP4 (`weight` U8 packed 2 values, `weight_scale` F8_E4M3 per 16 values, `weight_scale_2` F32 per tensor, `comfy_quant` marker), 48 layers, hidden 3840, vocab 262208, plus an unused SigLIP vision tower.

Block layout (per block): `attn1` video self-attention, `attn2` text cross-attention, `audio_attn1/2` the same for audio, `audio_to_video_attn` and `video_to_audio_attn`
(cross-modal), `ff` and `audio_ff` (GELU MLP, 4x). Every attention has q/k RMS norms and a per-head gate (`to_gate_logits`). Modulation tables:
`scale_shift_table` (9 x dim), `prompt_scale_shift_table`, `scale_shift_table_a2v_ca_*`.

## Workflow recap
1. Gemma encodes the prompt (optionally enhanced by `TextGenerateLTX2Prompt`), connectors turn it into video/audio text tokens.
2. Start image is resized (long edge), encoded by the video VAE, put in the first latent frame (`LTXVImgToVideoInplace`).
3. Stage 1: 768x512 latent video + latent audio, 8 distilled steps (sigmas `1.0 .. 0.0`), euler, CFG 1, distilled LoRA 0.5.
4. Spatial upscaler x2 on the video latent, stage 2: 3 steps (`0.85, 0.725, 0.4219, 0.0`).
5. Video VAE decode (tiled), audio VAE decode + vocoder, mux to mp4 (H.264 + AAC; the app must write MP4 itself: Media Foundation on Windows).

## Stages
| # | Deliverable | Verification |
|---|---|---|
| 0 | Tab, form, model roles, queue entry, engine stub (done) | tab visible, job reports a clear message |
| 1 | **Done.** `src/ltx/ltx_weights.*`: F32/BF16/FP8 (+scale)/NVFP4 (cuBLAS-tiled block scales) -> float32. `DopplersBakery ltx-weights <ckpt> <gemma>` vs `tools/ltx_ref.py` (ComfyUI code): 12 tensors, checksums identical | loader checksums equal ComfyUI |
| 2 | **Done.** `src/ltx/ltx_text.*`: Gemma 3 12B layer by layer (35 s for a 25-token prompt), per-token RMS + dual projection, 8-block video/audio connectors with registers. `DopplersBakery ltx-text` vs `tools/ltx_ref_text.py`: tokens identical, projection cosine 0.99974, video context 0.99990, audio context 0.99998 (ComfyUI runs bf16, this F16) | matches ComfyUI to bf16 precision |
| 3 | **Done (first version).** `src/ltx/ltx_vae.*`: decoder (3 x 2D convs per 3D conv, replicate padding in time, depth-to-space with first-frame drop, unpatchify) and the one-frame encoder. `DopplersBakery ltx-vae` vs `tools/ltx_ref_vae.py`: decode 3 latent frames to 17 frames 192x192 cosine 0.999997, encode 0.999993. Not tiled yet (needed for real sizes) | matches ComfyUI |
| 4 | **Done (first version).** `src/ltx/ltx_dit.*`: patchify, adaLN timesteps (per-frame video, prompt, cross-modal), split rope for video / audio / cross attention, gated attention, text cross-attention with adaLN, audio<->video exchange, output stage; one graph per block with its weights uploaded F16. `DopplersBakery ltx-dit` vs `tools/ltx_ref_dit.py` (ComfyUI LTXAVModel, 2 blocks, 48 video + 10 audio tokens): cosine 0.999988 video, 0.999994 audio. Explicit attention and host round trips per block: flash attention, layer streaming and speed are stage 7; per-frame timesteps (image-to-video) are not yet compared | one step output equals ComfyUI |
| 5 | **Done (first version).** `src/ltx/ltx_pipeline.*` (+ `ltx_upscaler.*`, LoRA in `ltx_weights.*`): Euler sampler with per-frame denoise mask, stage 1 (8 steps, half size) -> x2 upscaler (cosine 0.99998 vs ComfyUI) -> stage 2 (3 steps) -> decode; distilled LoRA merged on load (matches a torch merge, cosine 0.99999), merged weights cached as F16 in RAM. `DopplersBakery ltx-gen` made a coherent 256x256, 25-frame clip in 286 s (text-to-video, no start image). Missing: CRF-18 preprocess of the start image, i2v not yet run, audio not decoded, speed | video without sound |
| 6 | Audio VAE decode + vocoder, MP4 muxing (Media Foundation) | video with sound |
| 7 | Memory/speed work, UI polish, prompt enhancement, release | 4 s clip in a reasonable time on 12 GB |

References needed for exactness: the open-source `Lightricks/LTX-2` repository and ComfyUI's `comfy/ldm/lightricks` code; both are read, not copied wholesale.

