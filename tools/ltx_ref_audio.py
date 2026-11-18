# Reference for stage 6 (audio): ComfyUI's AudioVAE decoder + vocoder (BigVGAN + bandwidth extension), float32 on CPU, seeded normalized latent.
# Run from the ComfyUI folder:  python_embeded\python.exe -I tools\ltx_ref_audio.py <out dir> [T audio latent frames]
import sys, os, numpy as np, torch
C = r"c:\workenv\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI"
sys.path.insert(0, C); os.chdir(C)
import comfy.utils
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
T = int(sys.argv[2]) if len(sys.argv) > 2 else 12
sd, md = comfy.utils.load_torch_file(C + r"\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors", return_metadata=True)
asd = {}
for k, v in sd.items():
    if k.startswith("audio_vae."): asd["autoencoder." + k[len("audio_vae."):]] = v.float()
    elif k.startswith("vocoder."): asd[k] = v.float()
m = AudioVAE(metadata=md).float().eval()
miss, unexp = m.load_state_dict(asd, strict=False)
print("missing", [k for k in miss if "encoder" not in k][:5], "unexpected", unexp[:5])
z = torch.randn(1, 8, T, 16, generator=torch.Generator().manual_seed(5))
with torch.no_grad():
    lat = m.normalizer.denormalize(z)
    mel = m.autoencoder.decode(lat, target_shape=m.target_shape_from_latents(z.shape))
    wav = m.run_vocoder(mel)
print("mel", tuple(mel.shape), "wave", tuple(wav.shape), float(wav.abs().max()))
for n, a in (("au_z", z), ("au_mel", mel), ("au_wav", wav)): a.numpy().astype(np.float32).tofile(out + "\\" + n + ".bin")
