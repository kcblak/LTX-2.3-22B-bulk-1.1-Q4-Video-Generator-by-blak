import gc
import os
import sys
import json
import random
import tempfile
import glob
import traceback
import numpy as np
import subprocess
import psutil
import soundfile as sf
from PIL import Image

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch

# Detect GPU architecture once — used throughout to gate sm_60-specific workarounds
_GPU_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)   # P100 = sm_60; T4 = sm_75; A100 = sm_80; etc.
_IS_SM70_PLUS = (_GPU_SM[0] >= 7)

if _IS_SM60:
    # P100 (sm_60/Pascal) has NO native BF16 CUDA kernels and the current
    # PyTorch 2.4+ build dropped sm_60 support entirely.
    # Force FP16 to get as far as possible, and patch audio ops to run on CPU.
    os.environ["WGP_DTYPE"] = "fp16"
    print(f"  [GPU] sm_60 detected (P100) — FP16 mode + CPU audio patches enabled")
else:
    print(f"  [GPU] sm_{_GPU_SM[0]}{_GPU_SM[1]} detected — native CUDA mode, no patches needed")
import gradio as gr
from shared.utils.audio_video import save_video

# ==== GGUF EXTENSION HANDLER ====
# mmgp uses an extension handler system for non-safetensors formats.
# The full Wan2GP app registers this internally; for standalone scripts
# we must register it ourselves before any model loading happens.

def _register_gguf_handler():
    """Register the GGUF handler with mmgp's quant_router."""
    import shared.qtypes.gguf
    print("  [GGUF] ✅ Extension handler registered with mmgp (Wan2GP native)")

def _patch_ltx2_config_loading():
    """Patch _load_config_from_checkpoint to handle GGUF metadata errors."""
    import models.ltx2.ltx2 as ltx2_mod
    _original = ltx2_mod._load_config_from_checkpoint

    def _patched(path, fallback_config_path=None):
        from mmgp import quant_router
        if isinstance(path, (list, tuple)):
            path = path[0] if path else ""
        if not path:
            return {}
        try:
            _, metadata = quant_router.load_metadata_state_dict(path)
            if metadata:
                config_raw = metadata.get("config")
                if config_raw:
                    config = ltx2_mod._normalize_config(config_raw)
                    if config:
                        return config
        except Exception as e:
            print(f"  [GGUF Patch] Metadata read: {type(e).__name__}, using fallback config")
        if fallback_config_path and os.path.isfile(fallback_config_path):
            try:
                with open(fallback_config_path, "r", encoding="utf-8") as f:
                    config = ltx2_mod._normalize_config(json.load(f))
                    if config:
                        print(f"  [GGUF Patch] ✅ Config loaded from {os.path.basename(fallback_config_path)}")
                        return config
            except Exception:
                pass
        return {}

    ltx2_mod._load_config_from_checkpoint = _patched
    print("  [GGUF Patch] ✅ Config loading patched for GGUF")


# ==== GPU INFO ====
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ==== REGISTER GGUF + LOAD MODEL ====
print("\nLoading LTX-2.3 22B Distilled 1.1 (GGUF Q4_K_M)...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl

fl.set_checkpoints_paths(["models", "ckpts", "."])

from models.ltx2.ltx2_handler import family_handler

# Register GGUF handler + patch config loading BEFORE load_model
_register_gguf_handler()
_patch_ltx2_config_loading()


class _AudioEncoderP100Wrapper:
    """
    Wraps the audio encoder to fix cudaErrorNoKernelImageForDevice on P100.

    Root cause (ltx2.py line 1171-1176):
        audio_params = next(self.audio_encoder.parameters(), None)
        audio_device = audio_params.device   # returns 'cpu' under mmgp Profile 4
        mel = mel.to(device=audio_device, ...)  # mel goes to CPU
        audio_latent = self.audio_encoder(mel)  # mmgp moves encoder to CUDA
                                                # but mel is STILL on CPU
        -> Conv2d: weights on CUDA, input on CPU -> cudaErrorNoKernelImageForDevice

    Fix: intercept __call__ and move mel to CUDA FP16 BEFORE passing to encoder.
    All attribute access (sample_rate, mel_bins, parameters, etc.) is proxied.
    """
    def __init__(self, encoder):
        object.__setattr__(self, '_enc', encoder)

    def __call__(self, mel):
        if torch.cuda.is_available():
            mel = mel.to(
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=torch.float16,  # P100 (sm_60) has no BF16 CUDA kernels
            )
        return object.__getattribute__(self, '_enc')(mel)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_enc'), name)

    def __setattr__(self, name, value):
        if name == '_enc':
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, '_enc'), name, value)

    def __repr__(self):
        return f"_AudioEncoderP100Wrapper({object.__getattribute__(self, '_enc')!r})"

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}. Check download cell.")
print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

transformer_path = os.path.join("models", "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"{transformer_path} missing. Check download cell.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

# P100 (Pascal sm_60) has NO native BF16 support.
# Model weights loaded in FP16; but runtime activations (mel spectrogram) that
# flow through the BF16 autocast of the transformer are patched via CausalConv2d above.
# VAE_dtype = float32 for the video VAE (safer, audio VAE is patched separately).
MODEL_DTYPE = torch.float16
VAE_DTYPE   = torch.float16  # FP16 for T4: 65 TFLOPS vs FP32 8 TFLOPS = ~8x faster VAE decode

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=MODEL_DTYPE,
    VAE_dtype=VAE_DTYPE,
    text_encoder_filename=text_encoder_file,
)

# ==== Verify pipeline components ====
print("\n--- Pipeline Components ---")
for name, component in pipe.items():
    if component is not None:
        ctype = type(component).__name__
        if hasattr(component, 'parameters'):
            try:
                p = next(component.parameters())
                print(f"  {name}: {ctype} (dtype={p.dtype})")
            except StopIteration:
                print(f"  {name}: {ctype} (no params)")
        else:
            print(f"  {name}: {ctype}")
    else:
        print(f"  {name}: None")
sys.stdout.flush()

# ==== sm_60 (P100) Only: patch CausalConv2d BEFORE offload.profile() ====
# On sm_75+ (T4, A100, etc.) this is skipped — native CUDA handles everything.
# On sm_60 the entire PyTorch CUDA kernel set is unavailable (2.4+ dropped it),
# so F.pad + conv must run on CPU. This patch must be applied AFTER load_model
# (class importable) but BEFORE offload.profile (mmgp captures previous_method).
if _IS_SM60:
    try:
        import torch.nn.functional as _F
        from models.ltx2.ltx_core.model.audio_vae.causal_conv_2d import CausalConv2d as _CC2d

        def _cc2d_cpu_pad(self, x: torch.Tensor) -> torch.Tensor:
            if x.is_cuda:
                dev, dt = x.device, x.dtype
                x_cpu = x.detach().cpu().float()
                x_cpu = _F.pad(x_cpu, self.padding)
                w = self.conv.weight.detach().cpu().float()
                b = self.conv.bias.detach().cpu().float() if self.conv.bias is not None else None
                out = _F.conv2d(x_cpu, w, b,
                                self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups)
                return out.to(device=dev, dtype=dt)
            else:
                x = _F.pad(x, self.padding)
                return self.conv(x)

        _CC2d.forward = _cc2d_cpu_pad
        print("  [sm_60 Fix] ✅ CausalConv2d patched: pad+conv run on CPU")
    except Exception as _e:
        print(f"  [sm_60 Fix] ❌ Could not patch CausalConv2d: {_e}")
else:
    print("  [sm_60 Fix] ⏭️  Skipped (not sm_60)")

# ==== Apply mmgp Profile 4 ====
print("\nApplying mmgp Profile 4 with per-model budgets...")
sys.stdout.flush()

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.float16,
    budgets={
        # Budget 6000: ~9 min total (sweet spot — steps ~10s, VAE decode ~4 min)
        "transformer":       6000,
        "text_encoder":      1500,
        "video_encoder":     2000,
        "video_decoder":     3000,
        "audio_encoder":     1000,
        "audio_decoder":     1000,
        "vocoder":           500,
        "spatial_upsampler": 1500,
        "vae":               1000,
        "*":                 1000,
    },
)
offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Distilled 1.1 Text/Image-to-Video pipeline active.")
sys.stdout.flush()

# ==== HELPER FUNCTIONS ====
OUTPUT_DIR = "/kaggle/working/outputs"
_output_counter = 0

def list_outputs():
    if not os.path.isdir(OUTPUT_DIR):
        return []
    videos = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.mp4', '.mkv', '.webm'))]
    videos.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    return videos

def _get_next_output_path():
    global _output_counter
    _output_counter += 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"ltx_video_{_output_counter:04d}.mp4")

def get_resolution(base_res_str, aspect_ratio_str):
    base_resolutions = {"1080p": 1088, "720p": 704, "540p": 544, "480p": 480}
    ratios = {
        "16:9 Landscape": 16/9, "4:3 Standard": 4/3,
        "1:1 Square": 1.0, "3:4 Portrait": 3/4, "9:16 Portrait": 9/16,
    }
    base = base_resolutions.get(base_res_str, 704)
    ratio = ratios.get(aspect_ratio_str, 16/9)
    if ratio >= 1.0:
        height = base
        width = int(base * ratio)
    else:
        width = base
        height = int(base / ratio)
    return (width // 32) * 32, (height // 32) * 32

def get_vae_tile_size(height, width):
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    effective_vram = vram_mb / 1.5
    if effective_vram >= 24000: vae_config = 1
    elif effective_vram >= 8000: vae_config = 2
    else: vae_config = 3
    if max(height, width) > 480: vae_config += 1
    if vae_config <= 1: return 0
    elif vae_config == 2: return 512
    elif vae_config == 3: return 256
    return 128

def snap_to_ltx_frames(duration_sec: float, fps: float = 24.0, max_frames: int = 721) -> int:
    """Convert audio duration (seconds) to nearest valid LTX frame count.
    LTX distilled requires frames = 8k+1 (1, 9, 17, 25, ... 721, ...).
    Caps at max_frames (default 721 = 30s @ 24fps).
    """
    raw = duration_sec * fps
    # round to nearest 8k+1
    k = max(0, round((raw - 1) / 8))
    frames = 8 * k + 1
    frames = max(49, min(frames, max_frames))   # floor at 2s (49f), cap at max
    return int(frames)

@torch.inference_mode()
def Video_Generation(prompt, input_image_start, input_image_end, seed, duration_dropdown,
                     resolution_dropdown, aspect_ratio_dropdown,
                     guide_scale=3.0, num_steps=8, progress=gr.Progress()):
    try:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        progress(0, desc="Starting...")

        duration_map = {
            "2 Seconds (49 frames)":  49,
            "3 Seconds (73 frames)":  73,
            "5 Seconds (121 frames)": 121,
            "8 Seconds (193 frames)": 193,
            "10 Seconds (241 frames)": 241,
            "15 Seconds (361 frames)": 361,
        }
        frame_rate = 24.0
        num_frames = duration_map.get(duration_dropdown, 121)
        width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None
        image_end   = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")
        if input_image_end is not None:
            image_end = Image.open(input_image_end).convert("RGB")

        free_vram = torch.cuda.mem_get_info()[0] / 1024**3
        ram = psutil.virtual_memory()
        mode = "T2V" if image_start is None else ("I2V first+last" if image_end else "I2V start")
        print(f"\n{'='*60}")
        print(f"Generating [{mode}]: {width}x{height}, {num_frames} frames, seed={seed}")
        print(f"  VRAM free: {free_vram:.2f} GB | RAM free: {ram.available / 1024**3:.1f} GB")
        print(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  Guide scale: {guide_scale}")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Hardcode VAE tile size to 256 (matches audio notebook, prevents OOM/slowdowns at higher res)
        vae_tile_size = 256
        print(f"  VAE tile size: {vae_tile_size} (fixed)")

        total_steps = [8]
        current_step = [0]
        current_pass = [1]

        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            stage_name = (
                "Stage 1 (half-res)" if current_pass[0] == 1
                else "Stage 2 (full-res refine)" if current_pass[0] == 2
                else "Diffusion"
            )
            free_v = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | VRAM free: {free_v:.2f} GB")
            sys.stdout.flush()
            frac = current_step[0] / max(total_steps[0], 1)
            if current_pass[0] == 2:
                frac = 0.73 + 0.22 * frac
            else:
                frac = frac * 0.73
            progress(min(frac, 0.95), desc=f"{stage_name}: {current_step[0]}/{total_steps[0]}")

        _stage_labels = {
            "VAE Encoding": ("🇯️  VAE Encoding input frames...",   0.05),
            "VAE Decoding": ("🎦 VAE Decoding latents → frames...", 0.88),
            "Upsampling":   ("🔭 Spatial upsampling latents...",  0.83),
        }
        import time as _time
        _t = [_time.time()]

        def set_progress_status(status: str):
            dt = _time.time() - _t[0]; _t[0] = _time.time()
            label, frac = _stage_labels.get(status, (f"⏳ {status}...", 0.85))
            print(f"  [{status}] {label}  (+{dt:.1f}s)")
            sys.stdout.flush()
            progress(frac, desc=label)

        gen_kwargs = dict(
            input_prompt=prompt,
            image_start=image_start,
            height=height,
            width=width,
            frame_num=num_frames,
            fps=frame_rate,
            seed=seed,
            callback=cb,
            VAE_tile_size=vae_tile_size,
            input_video_strength=1.0,
            denoising_strength=1.0,
            guide_scale=float(guide_scale),
            sampling_steps=int(num_steps),
            guide_phases=2,
            n_prompt="",
            enhance_prompt=False,
            video_prompt_type="",
            audio_prompt_type="",
            set_progress_status=set_progress_status,
        )
        if image_end is not None:
            gen_kwargs["image_end"] = image_end

        print("  Diffusion starting → Stage 1 → spatial upsample → Stage 2 → VAE decode...")
        sys.stdout.flush()
        _t[0] = _time.time()

        result = ltx2_model.generate(**gen_kwargs)

        progress(0.97, desc="✅ Generation done — saving video...")
        print("  Pipeline complete.")
        sys.stdout.flush()

        if isinstance(result, dict):
            video_tensor = result.get("x")
            audio_data   = result.get("audio")
            audio_sr     = result.get("audio_sampling_rate", 24000)
        elif isinstance(result, tuple):
            video_tensor = result[0]
            audio_data   = result[1] if len(result) > 1 else None
            audio_sr     = result[2] if len(result) > 2 else 24000
        else:
            video_tensor = result
            audio_data, audio_sr = None, 24000

        if video_tensor is None or not torch.is_tensor(video_tensor):
            return None, f"❌ No video tensor. Got: {type(video_tensor)}"

        print(f"  Video tensor: {video_tensor.shape}, dtype={video_tensor.dtype}")
        video_tensor = video_tensor.cpu()
        gc.collect(); torch.cuda.empty_cache()

        out_path = _get_next_output_path()
        video_for_save = video_tensor.unsqueeze(0).float() / 127.5 - 1.0
        save_video(tensor=video_for_save, save_file=out_path, fps=frame_rate, normalize=True, value_range=(-1, 1))
        print(f"  ✅ Video saved: {out_path}")

        # ==== Mux native model audio (if generated) ====
        if audio_data is not None:
            try:
                import soundfile as sf
                audio_tmp = tempfile.mktemp(suffix=".wav")
                if isinstance(audio_data, np.ndarray):
                    audio_np = audio_data
                    if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
                        audio_np = audio_np.T
                    sf.write(audio_tmp, audio_np, int(audio_sr or 24000))
                elif torch.is_tensor(audio_data):
                    import torchaudio
                    cpu_audio = audio_data.cpu().float()
                    if cpu_audio.dim() == 1: cpu_audio = cpu_audio.unsqueeze(0)
                    if cpu_audio.dim() == 3: cpu_audio = cpu_audio.squeeze(0)
                    torchaudio.save(audio_tmp, cpu_audio, int(audio_sr or 24000))
                else:
                    raise ValueError(f"Unknown audio type: {type(audio_data)}")

                final_path = out_path.replace(".mp4", "_with_audio.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-i", out_path, "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", final_path
                ], check=True, capture_output=True)
                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    out_path = final_path
                    print(f"  ✅ Native audio muxed into output")
                else:
                    print(f"  ⚠️ Audio mux produced empty file, using video-only")
            except Exception as e:
                print(f"  ⚠️ Audio mux failed: {e}")

        del video_tensor, video_for_save
        gc.collect(); torch.cuda.empty_cache()
        progress(1.0, desc="Done!")
        return out_path, f"✅ Done! Seed: {seed} | {width}x{height} | {num_frames} frames"

    except Exception as e:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# ==== GRADIO UI (AIQUEST BRANDED) ====
CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1000px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); padding: 28px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(102,126,234,0.3); }
.brand-title { color: white; font-size: 2em; font-weight: 700; margin: 0 0 6px 0; }
.brand-subtitle { color: rgba(255,255,255,0.88); font-size: 1em; margin-bottom: 16px; }
.social-buttons { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.social-btn { padding: 10px 24px; border-radius: 8px; font-weight: 700; font-size: 15px; text-decoration: none; display: inline-block; color: white; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.social-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
.youtube-btn { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); }
.x-btn { background: linear-gradient(135deg, #000000 0%, #333333 100%); }
button.primary { background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#stop-btn { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
#clear-btn { background: linear-gradient(135deg, #6b7280 0%, #374151 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.footer { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb; color: #6b7280; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  BULK PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
import csv, io, zipfile, shutil, time as _time, threading

_bulk_stop_flag = threading.Event()

def _parse_bulk_csv(csv_text: str):
    """Parse CSV with columns: prompt, start_image, end_image (optional),
    duration, resolution, aspect_ratio, seed, guide_scale, steps.
    Returns list of dicts."""
    jobs = []
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    # normalise header names (strip spaces, lower)
    for row in reader:
        row = {k.strip().lower(): v.strip() for k, v in row.items()}
        jobs.append({
            "prompt":       row.get("prompt", ""),
            "start_image":  row.get("start_image", row.get("start image", "")),
            "end_image":    row.get("end_image",   row.get("end image",   "")),
            "duration":     row.get("duration",    "5 Seconds (121 frames)"),
            "resolution":   row.get("resolution",  "720p"),
            "aspect_ratio": row.get("aspect_ratio", row.get("aspect ratio", "16:9 Landscape")),
            "seed":         int(row.get("seed", -1)),
            "guide_scale":  float(row.get("guide_scale", row.get("guide scale", 3.0))),
            "steps":        int(row.get("steps", 8)),
        })
    return jobs

def _extract_images_zip(zip_path: str, dest_dir: str):
    """Unzip image archive into dest_dir, return {filename: full_path} map."""
    os.makedirs(dest_dir, exist_ok=True)
    img_map = {}
    if zip_path and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    zf.extract(name, dest_dir)
                    base = os.path.basename(name)
                    img_map[base] = os.path.join(dest_dir, name)
    return img_map

def run_bulk_queue(csv_file, zip_file,
                   default_duration, default_resolution,
                   default_aspect, default_guide, default_steps,
                   progress=gr.Progress()):
    """Main bulk runner called by the Gradio button."""
    global _bulk_stop_flag
    _bulk_stop_flag.clear()

    if csv_file is None:
        return None, "❌ Please upload a CSV file."

    # ── Read CSV ──────────────────────────────────────────────────────────────
    try:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            csv_text = f.read()
        jobs = _parse_bulk_csv(csv_text)
    except Exception as e:
        return None, f"❌ CSV parse error: {e}"

    if not jobs:
        return None, "❌ No jobs found in CSV."

    # ── Extract image zip ─────────────────────────────────────────────────────
    img_dir = "/kaggle/working/bulk_images"
    img_map = _extract_images_zip(zip_file, img_dir) if zip_file else {}

    total  = len(jobs)
    passed = []
    failed = []
    log_lines = [f"🚀 Starting {total} job(s)…\n"]

    for idx, job in enumerate(jobs):
        if _bulk_stop_flag.is_set():
            log_lines.append("🛑 Stopped by user.")
            break

        pct = idx / total
        progress(pct, desc=f"Job {idx+1}/{total}: {job['prompt'][:50]}…")
        log_lines.append(f"\n[{idx+1}/{total}] {job['prompt'][:80]}")

        # resolve image paths
        def _resolve(name):
            if not name:
                return None
            # absolute or relative to /kaggle/working
            for candidate in [name,
                               os.path.join("/kaggle/working", name),
                               img_map.get(os.path.basename(name), "")]:
                if candidate and os.path.exists(candidate):
                    return candidate
            return None

        start_path = _resolve(job["start_image"])
        end_path   = _resolve(job["end_image"])

        # apply per-job overrides or fall back to UI defaults
        duration  = job["duration"]    or default_duration
        res       = job["resolution"]  or default_resolution
        aspect    = job["aspect_ratio"]or default_aspect
        gscale    = job["guide_scale"] or default_guide
        steps     = job["steps"]       or default_steps
        seed      = job["seed"]

        try:
            out_path, status = Video_Generation(
                prompt              = job["prompt"],
                input_image_start   = start_path,
                input_image_end     = end_path,
                seed                = seed,
                duration_dropdown   = duration,
                resolution_dropdown = res,
                aspect_ratio_dropdown = aspect,
                guide_scale         = gscale,
                num_steps           = steps,
            )
            passed.append(out_path)
            log_lines.append(f"  ✅ {status}")
        except Exception as e:
            failed.append(str(e))
            log_lines.append(f"  ❌ Error: {e}")

        gc.collect(); torch.cuda.empty_cache()

    progress(1.0, desc="Done!")
    summary = (f"\n\n{'═'*50}\n"
               f"✅ Completed: {len(passed)}  ❌ Failed: {len(failed)}  "
               f"Total: {total}\n{'═'*50}")
    log_lines.append(summary)

    # Clean up extracted images to save space
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)

    last_video = passed[-1] if passed else None
    return last_video, "\n".join(log_lines)

def stop_bulk():
    _bulk_stop_flag.set()
    return "🛑 Stop signal sent — current job will finish, then queue halts."


def run_bulk_queue_with_auto_delete(csv_file, zip_file,
                                    default_duration, default_resolution,
                                    default_aspect, default_guide, default_steps,
                                    progress=gr.Progress()):
    """Bulk runner with auto-download and delete after each video."""
    global _bulk_stop_flag
    _bulk_stop_flag.clear()

    if csv_file is None:
        return None, "❌ Please upload a CSV file."

    try:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            csv_text = f.read()
        jobs = _parse_bulk_csv(csv_text)
    except Exception as e:
        return None, f"❌ CSV parse error: {e}"

    if not jobs:
        return None, "❌ No jobs found in CSV."

    img_dir = "/kaggle/working/bulk_images"
    img_map = _extract_images_zip(zip_file, img_dir) if zip_file else {}

    total = len(jobs)
    passed = []
    failed = []
    log_lines = [f"🚀 Starting {total} job(s)...\n"]
    all_output_paths = []

    for idx, job in enumerate(jobs):
        if _bulk_stop_flag.is_set():
            log_lines.append("🛑 Stopped by user.")
            break

        pct = idx / total
        progress(pct, desc=f"Job {idx+1}/{total}: {job['prompt'][:50]}…")
        log_lines.append(f"\n[{idx+1}/{total}] {job['prompt'][:80]}")

        def _resolve(name):
            if not name:
                return None
            for candidate in [name,
                               os.path.join("/kaggle/working", name),
                               img_map.get(os.path.basename(name), "")]:
                if candidate and os.path.exists(candidate):
                    return candidate
            return None

        start_path = _resolve(job["start_image"])
        end_path = _resolve(job["end_image"])

        duration = job["duration"] or default_duration
        res = job["resolution"] or default_resolution
        aspect = job["aspect_ratio"] or default_aspect
        gscale = job["guide_scale"] or default_guide
        steps = job["steps"] or default_steps
        seed = job["seed"]

        try:
            out_path, status = Video_Generation(
                prompt=job["prompt"],
                input_image_start=start_path,
                input_image_end=end_path,
                seed=seed,
                duration_dropdown=duration,
                resolution_dropdown=res,
                aspect_ratio_dropdown=aspect,
                guide_scale=gscale,
                num_steps=steps,
            )
            
            if out_path and os.path.exists(out_path):
                passed.append(out_path)
                all_output_paths.append(out_path)
                log_lines.append(f"  ✅ Generated: {os.path.basename(out_path)}")
                
                if GDRIVE_AVAILABLE:
                    try:
                        file_id = upload_video_to_gdrive(GDRIVE, out_path, GDRIVE_FOLDER_ID)
                        if file_id:
                            log_lines.append(f"  ✅ Backed up to Google Drive: {file_id}")
                    except Exception as e:
                        log_lines.append(f"  ⚠️  Google Drive upload error: {e}")
                else:
                    log_lines.append(f"  ⚠️  Google Drive not available - file retained for download")
            else:
                failed.append(job["prompt"])
                log_lines.append(f"  ❌ {status}")
        except Exception as e:
            failed.append(str(e))
            log_lines.append(f"  ❌ Error: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    progress(1.0, desc="✅ Batch complete!")
    summary = (f"\n\n{'═'*50}\n"
               f"✅ Completed: {len(passed)}  ❌ Failed: {len(failed)}  "
               f"Total: {total}\n{'═'*50}")
    log_lines.append(summary)

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)

    return None, "\n".join(log_lines)


def delete_selected_output(selected):
    if not selected:
        return list_outputs(), "❌ No video selected."
    path = os.path.join(OUTPUT_DIR, selected)
    if os.path.exists(path):
        try:
            os.remove(path)
            return list_outputs(), f"✅ Deleted {selected}"
        except Exception as e:
            return list_outputs(), f"❌ Delete failed: {e}"
    return list_outputs(), "❌ File not found."


def delete_all_outputs():
    if not os.path.isdir(OUTPUT_DIR):
        return [], "✅ No outputs to delete."
    for f in os.listdir(OUTPUT_DIR):
        if f.lower().endswith(('.mp4', '.mkv', '.webm')):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except Exception:
                pass
    return list_outputs(), "✅ All output videos deleted."


with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX-2.3 22B Video Generator | AIQUEST") as demo:
    gr.HTML('<div class="brand-header"><div class="brand-title">🎬 LTX-2.3 22B Distilled 1.1 Q4 — Video Generator</div><div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> | Kaggle T4 GPU</div><div class="social-buttons"><a href="https://youtube.com/@aiquestacademy" target="_blank" class="social-btn youtube-btn">▶️ Subscribe on YouTube</a><a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a></div></div>')

    with gr.Tabs():
        # ════════════════════════════════════════════════════════════════════════════════════
        # TAB 1: SINGLE VIDEO GENERATION
        # ════════════════════════════════════════════════════════════════════════════════════
        with gr.TabItem("🎬 Single Video", id="single"):
            gr.Markdown(
                "**Two-stage distilled 1.1 pipeline** (8 steps half-res → 2× upscale → 3 steps full-res) | Q4_K_M ~17.8GB\n\n"
                "💡 **Tip:** Leave image inputs empty for pure Text-to-Video. Add a start image for Image-to-Video."
            )

            with gr.Column():
                prompt = gr.Textbox(
                    label="🎨 Prompt", lines=3,
                    placeholder="A majestic eagle soaring over snowy mountain peaks at golden hour, cinematic, 4K..."
                )

                with gr.Accordion("🖼️ Image to Video (Optional)", open=False):
                    with gr.Row():
                        input_image_start = gr.Image(type="filepath", label="🎬 Start Frame (First Frame)")
                        input_image_end   = gr.Image(type="filepath", label="🎬 End Frame (Last Frame — Optional)")
                    gr.Markdown(
                        "*• **Start Frame only** → Image-to-Video (model generates from your image)*\n"
                        "*• **Both frames** → First+Last frame interpolation (model generates in-between)*\n"
                        "*• **Neither** → Pure Text-to-Video*"
                    )

                with gr.Row():
                    seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
                    duration_dropdown = gr.Dropdown(
                        label="⏱️ Duration",
                        choices=[
                            "2 Seconds (49 frames)",
                            "3 Seconds (73 frames)",
                            "5 Seconds (121 frames)",
                            "8 Seconds (193 frames)",
                            "10 Seconds (241 frames)",
                            "15 Seconds (361 frames)",
                        ],
                        value="5 Seconds (121 frames)",
                    )

                with gr.Row():
                    resolution_dropdown = gr.Dropdown(
                        label="📐 Base Resolution",
                        choices=["1080p", "720p", "540p", "480p"],
                        value="720p",
                    )
                    aspect_ratio_dropdown = gr.Dropdown(
                        label="📏 Aspect Ratio",
                        choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                        value="16:9 Landscape",
                    )

                guide_scale = gr.Slider(
                    label="🎯 Prompt Strength (guide_scale)",
                    minimum=1.0, maximum=8.0, step=0.5, value=3.0,
                    info="3.0 = optimal for T2V | 4.0+ = strong prompt | 1–2 = free generation",
                )
                num_steps = gr.Slider(
                    label="⚡ Diffusion Steps",
                    minimum=2, maximum=8, step=1, value=8,
                    info="6 = faster | 8 = best quality (default)",
                )

                with gr.Row():
                    gen_btn   = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_id="gen-btn")
                    stop_btn  = gr.Button("🛑 Stop",            variant="secondary", size="lg", elem_id="stop-btn")
                    clear_btn = gr.Button("🗑️ Clear",           variant="secondary", size="lg", elem_id="clear-btn")
                video_out  = gr.Video(label="🎥 Output")
                status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

                with gr.Accordion("🗂️ Output Manager", open=False):
                    gr.Markdown(f"**🔗 Google Drive Status:** {'✅ ENABLED - Videos will be backed up before deletion' if GDRIVE_AVAILABLE else '⚠️ DISABLED - Videos will be deleted without backup'}")
                    
                    refresh_outputs_btn = gr.Button("🔄 Refresh Outputs")
                    outputs_dropdown = gr.Dropdown(
                        label="Generated Videos",
                        choices=[],
                        interactive=True
                    )
                    
                    with gr.Row():
                        backup_btn = gr.Button("📤 Backup to Google Drive", variant="primary")
                        delete_output_btn = gr.Button("🗑️ Delete with Backup", variant="stop")
                    
                    with gr.Row():
                        backup_all_btn = gr.Button("📤 Backup All to Drive")
                        delete_all_btn = gr.Button("⚠️ Delete All (after backup)", variant="stop")
                    
                    delete_status = gr.Textbox(label="Status", interactive=False)

                    refresh_outputs_btn.click(
                        fn=list_outputs,
                        outputs=[outputs_dropdown]
                    )
                    
                    backup_btn.click(
                        fn=backup_video_to_gdrive,
                        inputs=[outputs_dropdown],
                        outputs=[delete_status]
                    )
                    
                    backup_all_btn.click(
                        fn=backup_all_videos_to_gdrive,
                        outputs=[delete_status]
                    )

                    delete_output_btn.click(
                        fn=delete_selected_with_gdrive_backup,
                        inputs=[outputs_dropdown],
                        outputs=[outputs_dropdown, delete_status]
                    )

                    delete_all_btn.click(
                        fn=delete_all_with_gdrive_backup,
                        outputs=[outputs_dropdown, delete_status]
                    )

                gen_event = gen_btn.click(
                    fn=Video_Generation,
                    inputs=[prompt, input_image_start, input_image_end, seed, duration_dropdown,
                            resolution_dropdown, aspect_ratio_dropdown, guide_scale, num_steps],
                    outputs=[video_out, status_out],
                )
                stop_btn.click(fn=None, cancels=[gen_event])
                clear_btn.click(
                    fn=lambda: (None, None, None, "", -1),
                    outputs=[input_image_start, input_image_end, video_out, prompt, seed],
                )

        # ════════════════════════════════════════════════════════════════════════════════════
        # TAB 2: BULK QUEUE GENERATION
        # ════════════════════════════════════════════════════════════════════════════════════
        with gr.TabItem("📦 Bulk Queue", id="bulk"):
            gr.Markdown(
                "### 🚀 Bulk Video Generation\n\n"
                "**How to use:**\n"
                "1. **Create a CSV** with columns: `prompt, start_image, end_image, duration, resolution, aspect_ratio, seed, guide_scale, steps`\n"
                "2. **Zip your images** (PNG/JPG) with filenames matching the CSV\n"
                "3. **Upload CSV + ZIP**, set defaults for missing values\n"
                "4. **Click ▶️ Run Bulk Queue** → Videos auto-generate and auto-delete\n\n"
                "**Example CSV:**\n"
                "```\n"
                "prompt,start_image,end_image,duration,seed\n"
                "Sunset over ocean,sunset_start.jpg,sunset_end.jpg,5 Seconds (121 frames),-1\n"
                "City timelapse,city.png,,3 Seconds (73 frames),42\n"
                "```"
            )

            with gr.Row():
                bulk_csv = gr.File(label="📄 CSV Manifest", file_types=[".csv"])
                bulk_zip = gr.File(label="🗜️ Images ZIP", file_types=[".zip"])

            gr.Markdown("**Default Settings** *(for rows with missing values):*")
            with gr.Row():
                bulk_dur = gr.Dropdown(
                    label="Duration",
                    choices=["2 Seconds (49 frames)","3 Seconds (73 frames)","5 Seconds (121 frames)","8 Seconds (193 frames)","10 Seconds (241 frames)","15 Seconds (361 frames)"],
                    value="5 Seconds (121 frames)"
                )
                bulk_res = gr.Dropdown(
                    label="Resolution",
                    choices=["1080p","720p","540p","480p"],
                    value="720p"
                )
                bulk_asp = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=["16:9 Landscape","4:3 Standard","1:1 Square","3:4 Portrait","9:16 Portrait"],
                    value="16:9 Landscape"
                )

            with gr.Row():
                bulk_guide = gr.Slider(label="Guide Scale", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                bulk_step = gr.Slider(label="Steps", minimum=2, maximum=8, step=1, value=8)

            with gr.Row():
                bulk_btn = gr.Button("▶️ Run Bulk Queue", variant="primary", size="lg")
                bulk_stp = gr.Button("🛑 Stop", variant="stop", size="lg")

            bulk_log = gr.Textbox(label="📋 Process Log", lines=15, interactive=False, placeholder="Processing logs appear here...")

            bulk_event = bulk_btn.click(
                fn=run_bulk_queue_with_auto_delete,
                inputs=[bulk_csv, bulk_zip, bulk_dur, bulk_res, bulk_asp, bulk_guide, bulk_step],
                outputs=[bulk_log],
            )
            bulk_stp.click(fn=stop_bulk, outputs=[bulk_log])

    gr.HTML('<div class="footer"><p style="font-size: 16px; margin: 5px 0;">🎬 Created by <strong>AIQuest Academy</strong></p><p style="font-size: 14px; margin: 5px 0; color: #9ca3af;">Free &amp; Open Source | LTX-2.3 22B Distilled 1.1 Q4_K_M | Kaggle T4 GPU</p><p style="font-size: 13px; margin: 10px 0;"><a href="https://youtube.com/@aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">YouTube</a> | <a href="https://x.com/aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">X (Twitter)</a> | <a href="https://aiquest.site" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">aiquest.site</a></p></div>')

print("\nLaunching Gradio...")
sys.stdout.flush()

demo.queue()
demo.launch(share=True, inline=False, debug=True, show_error=True, max_threads=1, ssr_mode=False)
