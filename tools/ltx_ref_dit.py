# Reference for stage 4 (the audio+video transformer): ComfyUI's LTXAVModel with the first N blocks, float32 on CPU, small seeded inputs.
# Run from the ComfyUI folder:  python_embeded\python.exe -I tools\ltx_ref_dit.py <ref dir> [N blocks] [F h w] [audio tokens]
import sys, os, json, numpy as np, torch
from safetensors import safe_open
C = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
sys.path.insert(0, C); os.chdir(C)
import comfy.ops, comfy.model_detection
from comfy.ldm.lightricks.av_model import LTXAVModel
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
F, h, w = (int(v) for v in sys.argv[3:6]) if len(sys.argv) > 5 else (3, 4, 4)
TA = int(sys.argv[6]) if len(sys.argv) > 6 else 10
P = "model.diffusion_model."
CK = C + r"\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors"
sd = {}
with safe_open(CK, "pt") as f:
    md = f.metadata(); keys = list(f.keys())
    for k in keys:
        if not k.startswith(P) or k.endswith(".weight_scale") or k.endswith(".input_scale"): continue
        if ".transformer_blocks." in k and int(k.split(".transformer_blocks.")[1].split(".")[0]) >= N: continue
        t = f.get_tensor(k)
        if t.dtype == torch.float8_e4m3fn: t = t.float() * f.get_tensor(k[:-len("weight")] + "weight_scale").float()
        sd[k] = t.float()
print("loaded", len(sd), "tensors")
cfg = json.loads(md["config"])["transformer"]
dc = comfy.model_detection.detect_unet_config(sd, P, metadata=md)
print("detected", {k: v for k, v in dc.items() if k in ("num_layers", "image_model", "cross_attention_adaln", "apply_gated_attention")})
dc = dict(dc); dc.pop("image_model", None); dc["num_layers"] = N
m = LTXAVModel(**dc, operations=comfy.ops.disable_weight_init, device="cpu", dtype=torch.float32).eval()
miss, unexp = m.load_state_dict({k[len(P):]: v for k, v in sd.items()}, strict=False)
print("missing", [k for k in miss][:6], "unexpected", [k for k in unexp][:6])
g = torch.Generator().manual_seed(1)
vx = torch.randn(1, 128, F, h, w, generator=g); ax = torch.randn(1, 8, TA, 16, generator=g)
vctx = np.fromfile(out + r"\video_ctx.bin", dtype=np.float32).reshape(1, -1, 4096); actx = np.fromfile(out + r"\audio_ctx.bin", dtype=np.float32).reshape(1, -1, 2048)
ctx = torch.cat([torch.from_numpy(vctx), torch.from_numpy(actx)], dim=-1)
t = torch.tensor([0.7])
with torch.no_grad():
    o = m([vx, ax], t, ctx, None, frame_rate=24, transformer_options={}, a_timestep=t.clone(), audio_length=TA)
vo, ao = o
print("out", tuple(vo.shape), tuple(ao.shape), float(vo.abs().max()), float(ao.abs().max()))
for n, a in (("dit_vx", vx), ("dit_ax", ax), ("dit_vo", vo), ("dit_ao", ao), ("dit_ctx", ctx)): a.detach().numpy().astype(np.float32).tofile(out + "\\" + n + ".bin")
