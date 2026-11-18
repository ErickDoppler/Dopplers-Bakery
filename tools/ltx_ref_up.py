# Reference for the x2 spatial latent upscaler: ComfyUI's LatentUpsampler (float32, CPU) on a seeded normalized latent.
# Run from the ComfyUI folder:  python_embeded\python.exe -I tools\ltx_ref_up.py <out dir> [T h w]
import sys, os, json, numpy as np, torch
from safetensors import safe_open
C = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
sys.path.insert(0, C); os.chdir(C)
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
T, h, w = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (3, 4, 4)
UP = C + r"\models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
with safe_open(UP, "pt") as f:
    cfg = json.loads(f.metadata()["config"]); m = LatentUpsampler.from_config(cfg).float().eval()
    m.load_state_dict({k: f.get_tensor(k).float() for k in f.keys()})
with safe_open(C + r"\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors", "pt") as f:
    mean = f.get_tensor("vae.per_channel_statistics.mean-of-means").float().view(1, -1, 1, 1, 1); std = f.get_tensor("vae.per_channel_statistics.std-of-means").float().view(1, -1, 1, 1, 1)
z = torch.randn(1, 128, T, h, w, generator=torch.Generator().manual_seed(3))
with torch.no_grad(): y = (m(z * std + mean) - mean) / std
print("up", tuple(z.shape), "->", tuple(y.shape), float(y.abs().max()))
z.numpy().astype(np.float32).tofile(out + r"\up_z.bin"); y.numpy().astype(np.float32).tofile(out + r"\up_y.bin")
