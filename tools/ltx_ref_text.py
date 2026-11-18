# Reference for stage 2 (Gemma 3 text path): token ids, projected embeddings and connector outputs computed by ComfyUI's code.
# Run from the ComfyUI folder with its embedded python:  python_embeded\python.exe -I tools\ltx_ref_text.py "<prompt>" <out dir>
import sys, os, numpy as np, torch
C = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
sys.path.insert(0, C); os.chdir(C)
import comfy.sd, comfy.utils, comfy.ops
from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
prompt = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
GM = C + r"\models\text_encoders\gemma_3_12B_it_fp4_mixed.safetensors"; CK = C + r"\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors"
clip = comfy.sd.load_clip(ckpt_paths=[GM, CK], embedding_directory=None, clip_type=comfy.sd.CLIPType.LTXV)
tokens = clip.tokenize(prompt)
ids = [t[0] for t in tokens["gemma3_12b"][0]]
mask_start = next(i for i, t in enumerate(ids) if t != 0)      # left padding with 0
real = ids[mask_start:]; print("tokens", len(real), real[:12])
np.array(real, dtype=np.int32).tofile(out + r"\ids.bin")
with torch.no_grad(): cond = clip.encode_from_tokens(tokens)   # [1, n, 4096 + 2048] dual-linear projection, before the connectors
cond = cond.float(); print("cond", tuple(cond.shape)); cond.detach().cpu().numpy().astype(np.float32).tofile(out + r"\proj.bin")
sd = comfy.utils.load_torch_file(CK)
def conn(prefix, heads, hd, x):
    m = Embeddings1DConnector(attention_head_dim=hd, num_attention_heads=heads, num_layers=8, split_rope=True, double_precision_rope=True, apply_gated_attention=True,
                              num_learnable_registers=128, operations=comfy.ops.disable_weight_init, dtype=torch.float32, device="cpu")
    m.load_state_dict({k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}, strict=True)
    with torch.no_grad(): return m(x.float().cpu())[0]
P = "model.diffusion_model."
vid = conn(P + "video_embeddings_connector.", 32, 128, cond[:, :, :4096]); aud = conn(P + "audio_embeddings_connector.", 32, 64, cond[:, :, 4096:])
print("video", tuple(vid.shape), "audio", tuple(aud.shape))
vid.numpy().astype(np.float32).tofile(out + r"\video_ctx.bin"); aud.numpy().astype(np.float32).tofile(out + r"\audio_ctx.bin")
