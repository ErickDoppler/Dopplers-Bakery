# Krea2 port notes (from ComfyUI source)

Sources read: `comfy/ldm/krea2/model.py`, `comfy/text_encoders/krea2.py`, `comfy/text_encoders/qwen3vl.py`,
`comfy/ldm/wan/vae.py`, `comfy/k_diffusion/sampling.py` (er_sde), `comfy/samplers.py` (beta), `comfy/model_sampling.py`.

## Pipeline (workflow krea2SimpleFine_v20AKV2, CFG 1 so the negative prompt is unused)
1. Prompt = pre_prompt + prompt, wrapped in the Qwen-Image template, tokenized (Qwen2 BPE, 151936 vocab).
2. Qwen3-VL-4B text tower (36 layers, hidden 2560, 32 heads / 8 KV, head_dim 128, theta 5e6, eps 1e-6).
   Collect hidden_states[2,5,8,...,35] (12 taps), drop the template prefix (up to and incl. "user\n"),
   result (seq, 12, 2560).
3. DiT `SingleStreamDiT`: features 6144, 48 heads / 12 KV, head_dim 128, 28 blocks, patch 2, 16 channels, SwiGLU mult 4.
   - `txtfusion`: 2 layerwise blocks over the 12-layer axis (batch = seq), `projector` Linear(12->1), 2 refiner blocks over seq.
   - `txtmlp`: RMSNorm -> Linear(2560,6144) -> GELU(tanh) -> Linear(6144,6144).
   - `tmlp(timestep_embedding(t,256))` -> `tproj` (GELU, Linear 6144->36864) = per-block modulation vector.
   - tokens = [text; image]; 3-axis RoPE ids: text (0,0,0), image (0,row,col); axes [headdim-12*(hd/16), 6*(hd/16), 6*(hd/16)] = [32,48,48], theta 1000.
   - block: x += pregate * attn((1+prescale)*prenorm(x)+preshift); x += postgate * mlp((1+postscale)*postnorm(x)+postshift).
     modulation = `mod.lin` + tvec, chunk(6) = prescale,preshift,pregate,postscale,postshift,postgate.
   - attention: q,k,v,gate projections, per-head RMSNorm on q,k (`(1+scale)` weight), RoPE, GQA, out = wo(attn * sigmoid(gate)).
   - RMSNorm everywhere uses weight `(1 + scale)`.
   - `last`: (1+scale)*norm(x)+shift using `last.modulation.lin` + t (the *un-projected* t), Linear(6144->64).
4. Sampler: flow matching, `ModelSamplingFlux` shift 1.15, sigma(t) = shift*t/(1+(shift-1)*t) over 10000 steps;
   scheduler `beta` (alpha=beta=0.6); sampler `ER-SDE` (max_stage 3), denoised = x - v*sigma.
   Initial latent = sigma0 * noise (CONST noise scaling).
5. Latent: Wan2.1 format, 16 ch, 8x spatial; `process_out(z) = z*std + mean` before the VAE.
6. VAE `qwen_image_vae` = Wan2.1 VAE decoder, single frame. T=1 collapses every CausalConv3d to a 2D conv using the LAST temporal
   tap (`weight[:,:,2]`), temporal resampling is skipped, so the decoder is plain 2D: conv1, middle(res, attn, res), 4 stages
   of 3 res blocks + nearest 2x upsample + conv, head (RMS, SiLU, conv 96->3). Output in [-1,1].

## Weights
All three files are scaled-FP8 (`F8_E4M3` + `*.weight_scale` F32 + `*.comfy_quant` U8 json) for big matrices; norms/biases BF16.
Dequant = fp8 * weight_scale. Plan: convert to ggml Q8_0 at load (scale folded into block scales), ~1 byte/param.
LoRA (`redbookstyle_krea2`): `diffusion_model.<path>.lora_A/B.weight` rank 32, merged W += strength * B@A.

## Stages and status
| Stage | Content | Status |
|---|---|---|
| A1 | safetensors reader | done |
| A2 | sigma schedule (flux shift + beta) and ER-SDE | done, matches ComfyUI (diff <1e-6) |
| A3 | Qwen2 BPE tokenizer (vocab.json/merges.txt embedded in exe) | done, token ids identical on 4 test strings |
| B1 | Qwen3-VL text tower on ggml | done, 12 taps cos >= 0.9997 vs ComfyUI; load 8 s, encode 0.6 s |
| B2 | DiT on ggml (FP8 -> Q8_0, optional F16 via BAKERY_DIT_F16=1) | done: cos 0.994 vs ComfyUI fp32 (ComfyUI bf16 vs its own fp32: 0.9905); 64-92 s/step at 720x720 on CPU |
| B3 | VAE decoder (2D, tiled above 64x64 latent) | done: cos 0.999996 vs ComfyUI (rms err 0.001); 0.4 s for 320x384 on CUDA |
| C | sampling loop, LoRA merge, PNG writer, runImage | done: 512x512 10 steps in 35 s (19 s load) on CUDA, 720x720 2.2 s/step; LoRA merged on 256 layers |
Each B stage is checked against tensors dumped from ComfyUI's embedded Python.

## Compute
CPU first (12.5 GB fp8 -> ~13 GB Q8_0; 720x720 = 2025 image tokens, ~26 TFLOP/step). A CUDA/Vulkan ggml backend is a separate follow-up;
CUDA needs prebuilt ggml-cuda DLLs because gcc cannot drive nvcc.

## GPU backend (done)
The exe links the official llama.cpp release build b12282 (same commit as third_party/llama.cpp headers): `llama.dll`, `ggml.dll`,
`ggml-base.dll`, `ggml-cpu-*.dll` (picked per CPU), and optional `ggml-cuda.dll` (+ cudart/cublas 13.5) or `ggml-vulkan.dll`.
They are kept in `runtime/` (git-ignored, ~750 MB) and copied next to the exe by `build.bat`. `src/gpu/compute.cpp` picks CUDA > Vulkan > CPU
(`BAKERY_BACKEND=cpu|cuda|vulkan` overrides) and uses the ggml scheduler with weights in pinned host memory, so the 13 GB DiT streams to a 12 GB card.
RTX 3080 Ti, 720x720 DiT step: CPU 64 s, CUDA 2.2 s, Vulkan 1.6 s. Output verified on both (DiT cos 0.994, text encoder cos 0.9998).
`DopplersBakery backends` lists the devices.
