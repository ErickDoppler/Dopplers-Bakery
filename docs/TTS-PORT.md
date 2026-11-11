# Qwen3-TTS port notes (from qwen3-tts-comfyui)

Sources: `custom_nodes/qwen3-tts-comfyui/qwen_tts/{core/models/modeling_qwen3_tts.py, core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py, inference/qwen3_tts_model.py}`.
Models in `models/qwen-tts/`: `Qwen3-TTS-12Hz-1.7B-{CustomVoice,Base,VoiceDesign}` (each holds `model.safetensors` BF16 and `speech_tokenizer/model.safetensors` F32).
CustomVoice first (preset speakers + optional style instruction); Base (voice cloning, needs the codec encoder + speaker encoder) and VoiceDesign later.

## Pipeline (CustomVoice, non-streaming-text = false)
1. Text -> `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` (Qwen2 BPE, same tokenizer as Krea2, files in the model folder).
   Optional instruction -> `<|im_start|>user\n{instruct}<|im_end|>\n`, projected and put first.
2. Talker prefill sequence (all vectors 2048-d): [instruct embeds] + role (3 text tokens through `text_projection`) +
   (tts_pad x (n-2), tts_bos) + codec prefill [think/nothink, think_bos, (language id), think_eos, speaker embed, pad, bos][:-1] summed pairwise,
   then first text token + last codec embed (bos). Text embeds = `text_projection(text_embedding(id))` (2-layer MLP, SiLU).
3. Talker = 28-layer Qwen3 decoder (hidden 2048, 16 heads/8 KV, head_dim 128, q/k RMSNorm, rope theta 1e6; the interleaved MRoPE collapses to plain rope because all three position axes are equal), `codec_head` -> 3072 logits (group-0 code).
   Sampling: temp 0.9, top-k 50, repetition penalty 1.05, tokens 2048..3071 suppressed except eos 2150, eos blocked for the first 2 steps.
4. Per frame: code predictor (5 layers, hidden 1024, 16/8 heads) gets [talker last hidden, embed(code0)] through `small_to_mtp_projection`, then 15 sequential
   steps (`lm_head[i]`, `codec_embedding[i]`, temp 0.9 top-k 50) -> codes 1..15.
5. Next talker input = sum of the 16 code embeddings (+ next trailing text embedding, then tts_pad once the text is consumed).
6. Speech decoder (`speech_tokenizer`): RVQ decode (1 semantic + 15 acoustic codebooks, `embedding_sum / max(cluster_usage, 1e-5)`, 1x1 output projections) -> causal conv (512->1024, k3)
   -> 8-layer transformer (hidden 512, 16 heads x 64, sliding window 72, layer scales) -> 2x [ConvTranspose x2 + ConvNeXt] -> causal conv 1024->1536 k7
   -> 4 blocks [Snake, ConvTranspose (8,5,4,3), 3 residual units dilation 1,3,9] -> Snake -> conv k7 -> clamp [-1,1]. 24 kHz, 1920 samples per frame.
   Decoded in chunks of 300 frames with 25 frames of left context.

## Stages
| Stage | Content | Status |
|---|---|---|
| T0 | reference dump (greedy codes + wav) from ComfyUI's python | done (`scratchpad/ref/tts_*`) |
| T1 | speech decoder: codes -> wav | done: cos 0.99986 vs ComfyUI, 0.4 s for 3.7 s of audio on CUDA |
| T2 | talker + code predictor with KV cache, prompt builder, sampling | done: prompt embeddings cos 0.999999; teacher-forced argmax agreement 93.5% (code0) and 95.2% (sub-codes) vs ComfyUI bf16; 25 ms per frame on CUDA (3x real time) |
| T3 | WAV writer, `runTTS` wiring, speakers/language/instruct UI, seed | done: runTTS writes 24 kHz WAV; VoiceDesign model works through the instruct field, Base (cloning) refused with a message |
| T4 | Base (voice cloning): mel + ECAPA-TDNN speaker encoder, Mimi codec encoder, in-context prompt | done: speaker embedding cos 0.999997, codec embeddings cos 0.999997 (codec encoder runs on the CPU scheduler: CUDA rounding costs 0.2%), codes 98.2% identical vs ComfyUI fp32; clone of Ryan: speaker similarity 0.989 to the reference |
