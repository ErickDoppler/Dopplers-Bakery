# Stable Audio 3 port notes (from ComfyUI)

Sources: `comfy/ldm/audio/{dit.py, embedders.py, vae_sa3.py}`, `comfy/text_encoders/sa3.py`, `comfy/model_base.py` (StableAudio3),
`comfy/model_sampling.py`, `comfy/k_diffusion/sampling.py` (sample_lcm). Workflow: `AUDIO3-MEDIUM.json`.
Checkpoint `stable_audio_3_medium.safetensors` (F32, 8.6 GB: DiT 5.4 GB + audio VAE 3.2 GB), text encoder `t5gemma_b_b_ul2.safetensors`
(with the SentencePiece model embedded as the `spiece_model` tensor).

## Pipeline
1. Prompt -> (optional LLM prompt enhancer, Qwen3.5 2B in the workflow) -> T5Gemma encoder (12 layers, 768 wide, tanh logit soft-capping 50) -> n x 768 tokens.
2. Conditioning: tokens padded to 256 with `padding_embedding`, plus one "seconds total" token (expo Fourier features of seconds/384 -> Linear) = 257 x 768.
3. DiT (`dit1.0`, 24 blocks, dim 1536, 24 heads x 64): 256-channel latents, 4096 samples per latent (10.77 Hz at 44.1 kHz stereo);
   preprocess 1x1 conv (+residual), project_in, 64 memory tokens prepended, partial rope (first 32 of 64 dims, theta 10000);
   per block: shared adaLN (`to_scale_shift_gate` + global cond, `x * sigmoid(1 - gate)`), differential self-attention (q,k,v,q',k' - attn(q,k,v) minus attn(q',k',v)),
   differential cross-attention to the 257 text tokens, constant per-layer local embedding (text-to-audio has zero local conditioning), SiLU-GLU feed-forward.
   Output: drop memory tokens, project_out, postprocess 1x1 conv (+residual).
4. Sampling: flow model (`CONST`), shift 2.0, `simple` scheduler over a 1000-entry table, LCM sampler, 8 steps, CFG 1.
   Initial latent = sigma0 * noise; denoised = x - v*sigma; LCM step: x = sigma_next * noise + (1 - sigma_next) * denoised.
5. Audio VAE decoder (`SA3AudioVAE`): Linear 256->1536, resampling block (12 transformer layers, differential attention, DyT norms, 8 sinusoidal-FF blocks,
   sliding window +-17 tokens, 16 new tokens per latent) -> 512 channels = 2 channels x 256 samples per token -> stereo 44.1 kHz.

## Stages
| Stage | Content | Status |
|---|---|---|
| S1 | SentencePiece tokenizer + T5Gemma encoder | done: token ids identical, cos 0.999999 (CPU) / 0.99998 (CUDA) |
| S2 | DiT | done: cos 0.999999 vs ComfyUI fp32 on CPU, 0.99996 CUDA, 0.99993 Vulkan; 0.10 s per forward at 64 latents on CUDA |
| S3 | Audio VAE decoder (chunked, 128 latents + 16 overlap) | done: cos 0.99993 CPU / 0.9999 CUDA vs ComfyUI; chunked decode identical to full; 0.03 s for 40 latents on CUDA |
| S4 | LCM sampler, simple schedule (shift 2.0), `runAudio`, stereo 44.1 kHz WAV | done: 60 s of audio in 1.5 s of sampling+decode on CUDA (8.5 s with the 7 s model load); output spectrum and rhythm look like music |
| S5 | Prompt expansion: the workflow's per-mode system prompts (data/sa_prompts.json, embedded) run through the LLM engine (GGUF: Audio tab setting, else the LLM tab model); MP3 export | expansion done (Music, SFX, One-shot checked with Qwen3-4B-Instruct); MP3 export done (Windows Media Foundation encoder, no extra DLLs: Audio default MP3 V0 = 256 kbps, 128k, 320k; TTS optional 128 kbps), the Qwen3.5-2B safetensors cannot be used (hybrid Gated DeltaNet, no GGUF) |

Notes: the fused CUDA flash-attention crashes for T5Gemma's softcapped head-dim-64 attention without a mask, so that encoder uses plain attention.
