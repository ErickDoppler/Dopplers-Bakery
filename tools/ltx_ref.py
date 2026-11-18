# Reference values for `DopplersBakery ltx-weights`: dequantizes the same tensors with ComfyUI's own code.
# Run with ComfyUI's embedded python:  python_embeded\python.exe -I tools\ltx_ref.py
import torch
from safetensors import safe_open
import comfy_kitchen.backends.eager.quantization as q
from comfy_kitchen.float_utils import from_blocked
M = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI\models"
FILES = [(M + r"\checkpoints\ltx-2.3-22b-dev-fp8.safetensors",
          ["model.diffusion_model.transformer_blocks.20.attn1.to_q.weight", "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight", "model.diffusion_model.patchify_proj.weight",
           "model.diffusion_model.transformer_blocks.47.ff.net.2.weight", "model.diffusion_model.transformer_blocks.20.scale_shift_table", "vae.decoder.conv_in.conv.weight"]),
         (M + r"\text_encoders\gemma_3_12B_it_fp4_mixed.safetensors",
          ["model.layers.5.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.10.mlp.gate_proj.weight", "model.layers.47.self_attn.q_proj.weight", "model.embed_tokens.weight", "model.norm.weight"])]
for path, names in FILES:
    with safe_open(path, "pt") as f:
        keys = set(f.keys()); print(path.split("\\")[-1])
        for n in names:
            if n not in keys: print("  ", n, "missing"); continue
            w = f.get_tensor(n); b = n[:-7] if n.endswith(".weight") else n
            if b + ".comfy_quant" in keys and b"nvfp4" in bytes(f.get_tensor(b + ".comfy_quant").tolist()):
                v = q.dequantize_nvfp4(w, f.get_tensor(b + ".weight_scale_2"), f.get_tensor(b + ".weight_scale"), torch.float32); fmt = "nvfp4"
            elif w.dtype == torch.float8_e4m3fn and b + ".weight_scale" in keys:
                v = w.float() * f.get_tensor(b + ".weight_scale").float(); fmt = "fp8"
            else: v = w.float(); fmt = "none"
            v = v.flatten(); print("  %-70s %-8s sum=%.6g sumabs=%.6g first=%.6g %.6g %.6g %.6g" % (n, fmt, v.double().sum(), v.double().abs().sum(), *v[:4].tolist()))
