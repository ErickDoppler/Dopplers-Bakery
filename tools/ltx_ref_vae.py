# Reference for stage 3 (video VAE): decodes a seeded random latent with ComfyUI's VAE (float32, CPU) and, if given an image, encodes it.
# Run from the ComfyUI folder:  python_embeded\python.exe -I tools\ltx_ref_vae.py <out dir> [T h w]
import sys, os, numpy as np, torch
C = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
sys.path.insert(0, C); os.chdir(C)
import comfy.sd, comfy.utils
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
T, h, w = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (3, 6, 6)
sd, md = comfy.utils.load_torch_file(C + r"\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors", return_metadata=True)
vae = comfy.sd.VAE(sd={k[4:]: v for k, v in sd.items() if k.startswith("vae.")}, metadata=md)
m = vae.first_stage_model.float().cpu().eval()
g = torch.Generator().manual_seed(0)
z = torch.randn(1, 128, T, h, w, generator=g)
with torch.no_grad(): y = m.decode(z)
print("latent", tuple(z.shape), "->", tuple(y.shape), float(y.min()), float(y.max()))
z.numpy().astype(np.float32).tofile(out + r"\vae_z.bin"); y.numpy().astype(np.float32).tofile(out + r"\vae_y.bin")
# encoder: a deterministic 1-frame "image" in [-1, 1] (a smooth pattern plus noise), 64 x 64
x = torch.randn(1, 3, 1, 64, 64, generator=g).clamp(-1, 1) * 0.3 + torch.linspace(-0.6, 0.6, 64).view(1, 1, 1, 1, 64)
with torch.no_grad(): e = m.encode(x)
print("image", tuple(x.shape), "->", tuple(e.shape))
x.numpy().astype(np.float32).tofile(out + r"\vae_x.bin"); e.numpy().astype(np.float32).tofile(out + r"\vae_e.bin")
# debug: output of conv_in for the seeded latent (un-normalized), to bisect the decoder
with torch.no_grad():
    from comfy.ldm.lightricks.vae.causal_video_autoencoder import mark_conv3d_ended
    zz = m.per_channel_statistics.un_normalize(z); mark_conv3d_ended(m.decoder.conv_in); ci = m.decoder.conv_in(zz, causal=False)
ci.numpy().astype(np.float32).tofile(out + r"\vae_convin.bin"); print("convin", tuple(ci.shape))


