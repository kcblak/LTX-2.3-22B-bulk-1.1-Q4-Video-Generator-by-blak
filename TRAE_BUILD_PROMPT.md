# MASTER BUILD PROMPT FOR TRAE
Paste everything below this line into Trae as a single instruction.

---

You are building, from scratch, a complete, production-ready Kaggle notebook system called **"LTX-2.3 22B Distilled 1.1 Q4 — Video Generator"** for the GitHub repo:

`https://github.com/kcblak/LTX-2.3-22B-bulk-1.1-Q4-Video-Generator-by-blak`

This system generates AI videos (text-to-video and image-to-video) using the LTX-2.3 22B distilled model in GGUF Q4_K_M quantization, running on a free Kaggle T4 GPU (16GB VRAM, ~30GB RAM), via the Wan2GP inference engine and a Gradio web UI with a single-video tab and a bulk CSV-queue tab with Google Drive backup.

I am giving you the full context of a previous broken attempt at this system, including the exact failure that occurred, so you do not repeat it. Read the "KNOWN FAILURE MODES — DO NOT REPEAT" section carefully before writing any code.

## 1. REPOSITORY STRUCTURE TO CREATE

Create this exact structure in the repo root:

```
LTX-2.3-22B-bulk-1.1-Q4-Video-Generator-by-blak/
├── README.md
├── ltx-2-3-22b-bulk-1-1-q4-video-generator.ipynb   <- the ONLY notebook. Single source of truth.
├── run_ltx_t2v.py                                   <- reference copy only, regenerated at runtime by the notebook (see §6)
├── sample_jobs/
│   ├── jobs_template.csv
│   └── jobs_example.csv
└── .gitignore
```

Do **NOT** create a second, simplified, "Kaggle dataset"-dependent notebook (e.g. a `kaggle-ltx-notebook.ipynb` that copies `run_ltx_t2v.py` from `/kaggle/input/datasets/...`). That pattern is the exact thing that broke previously (see §0 below). There must be exactly **one notebook**, and it must be **fully self-contained**: every file it needs (the Python script, model downloads, etc.) is created or fetched by the notebook's own cells, never assumed to pre-exist in an attached Kaggle Dataset.

## 0. KNOWN FAILURE MODES — DO NOT REPEAT

A prior version of this project produced two divergent notebooks and ran into a real failure. Read this before writing anything.

**Failure #1 — torch version drift breaks the GPU compatibility patch.**
The notebook pins `torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1` (via `--index-url https://download.pytorch.org/whl/cu121`) specifically because PyTorch 2.4+ dropped CUDA kernel support for `sm_60` (Pascal/P100) GPUs, and the whole script has P100 BF16-CPU-fallback patches built around `torch==2.3.1` being the actual installed version. In the broken run, a **later pip install command silently upgraded torch to 2.12.0**, which:
- defeated the sm_60 compatibility patches entirely,
- produced dependency-resolver warnings (`torchvision 0.18.1+cu121 requires torch==2.3.1, but you have torch 2.12.0`),
- left the environment in an inconsistent state for the rest of the run.

**Fix you must implement:** after installing `mmgp`, `gradio`, `gguf`, `soundfile`, and `Wan2GP/requirements.txt`, add an explicit guard cell that re-asserts the torch pin and fails loudly (not silently continues) if any subsequent pip install changed the torch version. Concretely:
- Install torch/torchvision/torchaudio FIRST, exactly pinned, with `--index-url https://download.pytorch.org/whl/cu121`.
- Install all other dependencies with `pip install --no-deps` where the package would otherwise try to pull in a newer torch as a transitive dependency, OR pass `"torch==2.3.1"` again as a constraint via a `constraints.txt` file passed to every subsequent `pip install -c constraints.txt`.
- After all installs, add a verification cell:
  ```python
  import torch
  assert torch.__version__.startswith("2.3.1"), (
      f"torch was upgraded to {torch.__version__} by a later pip install — "
      f"this breaks the sm_60/P100 compatibility patches. "
      f"Re-run cell 1 (PyTorch install) and do NOT install any package that pulls a newer torch."
  )
  print(f"✅ torch version locked at {torch.__version__}")
  ```
- This assertion cell must run AFTER every pip install cell and BEFORE the model-loading cell. If it fails, the notebook should stop (raise), not continue into a broken model load.

**Failure #2 — a script-not-found crash from a "copy from Kaggle Dataset" pattern.**
A second, simplified notebook variant assumed `run_ltx_t2v.py` already existed in an attached Kaggle Dataset at `/kaggle/input/datasets/investorblak/ltx-batch/run_ltx_t2v.py`, and only copied it if present:
```python
script_src = '/kaggle/input/datasets/investorblak/ltx-batch/run_ltx_t2v.py'
if os.path.exists(script_src):
    shutil.copy(script_src, '/kaggle/working/run_ltx_t2v.py')
else:
    print('WARNING: Script not in dataset')
```
When that dataset wasn't attached (the normal case for a fresh "Run All" on a public notebook), the script was never written, and the launch cell crashed with:
```
python3: can't open file '/kaggle/working/run_ltx_t2v.py': [Errno 2] No such file or directory
```
**Fix you must implement:** the notebook must **always generate `run_ltx_t2v.py` itself**, in its own cell, using `%%writefile run_ltx_t2v.py` (or an equivalent `Path(...).write_text(...)` call). It must never depend on an externally attached Kaggle Dataset for the script itself. Models (the large `.gguf`/`.safetensors` files) may legitimately come from Hugging Face Hub downloads (that part of the original notebook is correct and should be kept), but the Python script must be authored inline in the notebook, every single run, with no conditional fallback path that can silently no-op.

**Failure #3 — fragile post-hoc string-surgery to inject the Bulk Queue tab.**
A previous version wrote the single-video Gradio UI first, then used a separate cell that ran a Python heredoc doing `text.replace(...)` on the raw source of `run_ltx_t2v.py` to retroactively wrap it in `gr.Tabs()` and splice in a bulk-queue tab, with manual re-indentation of every line via string matching (`if line and not line.startswith('        ')`). This is extremely fragile — it silently produces broken indentation or duplicate UI if cell run order changes, if emoji/text in markers don't match byte-for-byte, or if the notebook is re-run from a saved checkpoint.
**Fix you must implement:** write `run_ltx_t2v.py` ONCE, in ONE cell, with the full final UI (both tabs, all functions, Google Drive integration, etc.) already correctly structured and indented as static Python source in the `%%writefile` heredoc. Do not use any post-hoc string-replace patching cell. If you need to compose the script from logical sections, build it as one Python string in the notebook and write it out in one shot — never patch a previously-written file with regex/string-replace.

**Failure #4 — duplicated/conflicting function definitions across cells.**
The previous notebook defined `run_bulk_queue` in one cell and a near-duplicate `run_bulk_queue_with_auto_delete` in a later cell, and defined `list_outputs`/`OUTPUT_DIR` in two places with slightly different logic (timestamp-based filenames vs. an incrementing counter). This produces confusing behavior depending on which definition "wins" at runtime.
**Fix you must implement:** every function and global must be defined exactly once in the final `run_ltx_t2v.py`. Decide on ONE filename scheme (use an incrementing counter, `ltx_video_0001.mp4`, `ltx_video_0002.mp4`, ... — not Unix timestamps, since counters are more predictable for users matching outputs to bulk-CSV rows) and ONE bulk-runner function (the auto-delete-after-Drive-backup version, since that's the more complete feature set), and remove the redundant earlier version entirely.

## 2. ENVIRONMENT & HARDWARE TARGET

- Platform: Kaggle Notebooks, GPU accelerator **T4 x1** (16GB VRAM). Also support P100 (16GB, sm_60) gracefully via runtime GPU-capability detection — the script must detect compute capability at import time and branch behavior (see §6.4), but T4 is the primary target and the one actually exercised in testing.
- ~30GB system RAM, no swap available — never assume swap exists.
- Disk: `/kaggle/working` is the persistent, downloadable output area (limited size — keep this lean). `/kaggle/tmp` is larger scratch space for big model files, used via symlinks into `/kaggle/working` or `Wan2GP/models` so paths stay short while actual bytes live in the bigger area.

## 3. MODELS TO DOWNLOAD (Hugging Face Hub)

Primary transformer (GGUF Q4_K_M, distilled 1.1):
- Repo: `Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF`
- File: `LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf`
- Destination filename after download: `ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` (lowercase-prefixed, this exact casing is required by the loader code in §6)

Companion large files (from `DeepBeepMeep/LTX-2`):
- `ltx-2.3-22b-distilled-lora-384.safetensors`
- `ltx-2.3-22b_embeddings_connector.safetensors`
- `ltx-2.3-22b_text_embedding_projection.safetensors`
- `ltx-2.3-22b_vae.safetensors`

Smaller companion files (from `DeepBeepMeep/LTX-2`, downloaded directly into `Wan2GP/models`, no symlink needed):
- `ltx-2.3-22b_audio_vae.safetensors` (~107 MB)
- `ltx-2.3-22b_vocoder.safetensors` (~258 MB)
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (~996 MB)
- `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` (~262 MB)

Text encoder (Gemma, from `DeepBeepMeep/LTX-2`, folder `gemma-3-12b-it-qat-q4_0-unquantized`):
- `gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors`
- `added_tokens.json`, `chat_template.json`, `config_light.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`

Download logic requirements:
- Use `huggingface_hub.hf_hub_download`, downloading the big files into `/kaggle/tmp/models` and symlinking into `Wan2GP/models`, exactly as the working pattern below (this part was correct in the original and should be preserved):
  ```python
  import os
  from huggingface_hub import hf_hub_download

  REPO = 'Abiray/LTX-2.3-22B-DISTILLED-1.1-GGUF'
  COMPANION_REPO = 'DeepBeepMeep/LTX-2'
  MODEL_DIR = 'Wan2GP/models'
  TMP_DIR = '/kaggle/tmp/models'
  os.makedirs(MODEL_DIR, exist_ok=True)
  os.makedirs(TMP_DIR, exist_ok=True)
  ```
- Every download must be idempotent: check `os.path.exists(dest)` first and skip with a `✓ Already exists` message if so — this lets users re-run the notebook (e.g. after a kernel restart) without re-downloading tens of GB.
- After all downloads, clean any stray `.cache` directories under `MODEL_DIR`/`TMP_DIR` and print `df -h /kaggle/working /kaggle/tmp` so the user can see remaining disk space.
- Wrap every `hf_hub_download` call in a try/except that prints a clear, actionable error (e.g. "rate-limited — set HF_TOKEN as a Kaggle secret and re-run") rather than letting a bare `HTTPError` crash the whole cell with no remediation hint.

## 4. NOTEBOOK CELL-BY-CELL PLAN

Build the notebook with these cells, in this exact order. Each numbered item is one cell (markdown cells are explicitly noted; everything else is code).

1. **Markdown — title/banner.** Project title, one-line description, link to the GitHub repo, and the instruction: *"After running Cell 2 (PyTorch install), restart the kernel (Kernel → Restart Kernel) and then Run All from the top. Do not skip the restart — PyTorch installed via pip into an already-imported `torch` session will not take effect."*

2. **Markdown — "Step 1: Environment Setup".**

3. **Code — environment optimization.** RAM/disk reporting, `PYTORCH_CUDA_ALLOC_CONF` with `expandable_segments:True,garbage_collection_threshold:0.6`, `MALLOC_TRIM_THRESHOLD_=0`, `TOKENIZERS_PARALLELISM=false`. (Preserve this from the original — it was correct.)

4. **Markdown — "Step 2: Install PyTorch (pinned) and Clone Wan2GP".**

5. **Code — GPU check + clone + pinned PyTorch install.**
   - `nvidia-smi` check with a clear "go to Settings → Accelerator → GPU T4 x1" message on failure (don't hard-crash; let the rest of the notebook still attempt to run on CPU detection paths if you want graceful degradation, but warn loudly).
   - `git clone https://github.com/DeepBeepMeep/Wan2GP.git` (with `--depth 1`).
   - Install pinned PyTorch FIRST: `pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`.
   - Print: `"✅ PyTorch 2.3.1 installed. Now: Kernel → Restart Kernel, then Run All."`
   - This is the cell after which the user must restart the kernel, per the banner instruction.

6. **Markdown — "Step 3: Install Remaining Dependencies (run after kernel restart)".**

7. **Code — remaining dependency install + torch-pin guard (Failure #1 fix).**
   - Write a `constraints.txt` containing `torch==2.3.1`, `torchvision==0.18.1`, `torchaudio==2.3.1`.
   - `pip install --timeout 120 --retries 5 -q -r Wan2GP/requirements.txt -c constraints.txt`
   - `pip install --timeout 120 --retries 5 -q mmgp gradio gguf soundfile pydrive2 google-auth google-auth-oauthlib google-api-python-client -c constraints.txt`
   - Immediately after, run the torch-version assertion cell described in Failure #1 above. This must be in the SAME cell as the installs (not a separate later cell), so there is zero chance of someone running cells out of order and missing the check.

8. **Markdown — "Step 4: Download Models".**

9. **Code — model downloads.** Exactly per §3 above, fully idempotent, with disk-space reporting and try/except remediation messages.

10. **Markdown — "Step 5: Write the Inference + Gradio Script".**

11. **Code — `%%writefile run_ltx_t2v.py`, the ENTIRE script in one cell.** This is the single biggest cell and must contain the complete, final, correctly-indented script as described in §6. No follow-up patch cells. No duplicated functions. No second draft.

12. **Markdown — "Step 6: Google Drive Backup Setup (optional, for Bulk Queue auto-delete)".** Explain: if the user wants bulk-generated videos backed up to Google Drive before local auto-deletion, they should add a Kaggle Secret containing a GCP service-account JSON key (see §6.5 for exactly how this is wired up), name it `GDRIVE_SERVICE_ACCOUNT_JSON`, and set `GDRIVE_SHARED_DRIVE_ID` (also as a Kaggle Secret) if uploading to a Shared Drive. If neither secret is present, the notebook must run perfectly fine without Drive backup — it just skips that feature with a clear one-line status message, and bulk-queue auto-delete is disabled (videos are kept locally instead of being silently lost).

13. **Code — read Kaggle Secrets for Google Drive (if present) and write them to environment variables** that `run_ltx_t2v.py` reads at import time (see §6.5). Must not crash if the secrets are absent — `kaggle_secrets.UserSecretsClient` calls must be wrapped in try/except, defaulting `GDRIVE_AVAILABLE=False`.

14. **Markdown — "Step 7: Launch".** Explain that the public Gradio URL will appear in the cell output, and that the user should watch for it.

15. **Code — launch.** `!cd /kaggle/working && python -u run_ltx_t2v.py 2>&1`

16. **Markdown — footer/credits** (keep minimal, your branding, link back to the GitHub repo).

Do not add any cell that copies, patches, or conditionally regenerates `run_ltx_t2v.py` from anywhere other than cell 11. Do not add a "Step 4.5: Copy Script" cell that reads from a Kaggle Dataset path.

## 5. BULK CSV SCHEMA (must match exactly)

The CSV format used by the Bulk Queue tab has these exact columns, in this order, header row required:

```
prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guide_scale,steps
```

Column semantics:
- `prompt` (required, string): the text prompt. May be long/multi-line (must be properly CSV-quoted with embedded newlines — the parser must use Python's `csv` module, never naive `.split(',')`).
- `start_image` (optional, string): filename of the first/start frame image, matched against filenames inside the uploaded images ZIP. Empty = pure text-to-video.
- `end_image` (optional, string): filename of the last frame image for first+last-frame interpolation. Empty = either pure T2V (if `start_image` also empty) or start-image-only I2V.
- `duration` (optional, string): one of the exact dropdown strings: `"2 Seconds (49 frames)"`, `"3 Seconds (73 frames)"`, `"5 Seconds (121 frames)"`, `"8 Seconds (193 frames)"`, `"10 Seconds (241 frames)"`, `"15 Seconds (361 frames)"`. Falls back to the bulk-tab default if empty.
- `resolution` (optional, string): one of `"1080p"`, `"720p"`, `"540p"`, `"480p"`. Falls back to default if empty.
- `aspect_ratio` (optional, string): one of `"16:9 Landscape"`, `"4:3 Standard"`, `"1:1 Square"`, `"3:4 Portrait"`, `"9:16 Portrait"`. Falls back to default if empty.
- `seed` (optional, int): `-1` means random. Falls back to `-1` if empty/missing.
- `guide_scale` (optional, float): falls back to default (3.0) if empty.
- `steps` (optional, int): falls back to default (8) if empty.

The parser must:
- Read with `encoding='utf-8-sig'` (the sample `jobs.csv` has a UTF-8 BOM — confirmed from real user data, must be handled, not just hoped-for).
- Normalize header names: `.strip().lower()`, and accept both `start_image`/`start image` and `end_image`/`end image` as aliases (underscore and space variants), since users editing in Excel/Google Sheets sometimes auto-convert underscores to spaces in headers.
- Only `prompt` is truly required; every other column may be entirely absent from the CSV (not just empty-valued) without crashing — use `row.get(col, default)` throughout, never `row[col]`.
- Multi-line prompts (prompts containing embedded newlines, as in the real sample data which has prompts like full shot-by-shot screenplay breakdowns) must round-trip correctly through `csv.DictReader` — this works automatically as long as the file is properly quoted (Excel/Sheets do this correctly on export), so just make sure you're using `csv.DictReader(io.StringIO(csv_text))`, never manual line-splitting.

Create `sample_jobs/jobs_template.csv` with just the header row and one example row using short placeholder values, and `sample_jobs/jobs_example.csv` with 2-3 realistic example rows (short prompts, not multi-paragraph) so users have something to test with immediately.

## 6. `run_ltx_t2v.py` — FULL TECHNICAL SPECIFICATION

This is the script written by notebook cell 11. Build it as ONE coherent, correctly-indented Python file. Sections below map to logical regions of the file — write them in this order.

### 6.1 Imports & bootstrap
```python
import gc, os, sys, json, random, tempfile, glob, traceback, subprocess, threading, time, csv, io, zipfile, shutil
import numpy as np
import psutil
import soundfile as sf
from PIL import Image

WAN2GP_DIR = os.path.abspath("Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
assert torch.__version__.startswith("2.3.1"), (
    f"run_ltx_t2v.py started with torch=={torch.__version__}, expected 2.3.1. "
    f"Re-run the PyTorch install cell and restart the kernel."
)
```

### 6.2 GPU architecture detection (keep from original — this logic is sound)
```python
_GPU_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
_IS_SM60 = (_GPU_SM[0] == 6)        # P100 = sm_60
_IS_SM70_PLUS = (_GPU_SM[0] >= 7)   # T4 = sm_75, A100 = sm_80, etc.

if _IS_SM60:
    os.environ["WGP_DTYPE"] = "fp16"
    print("  [GPU] sm_60 detected (P100) — FP16 mode + CPU audio patches enabled")
else:
    print(f"  [GPU] sm_{_GPU_SM[0]}{_GPU_SM[1]} detected — native CUDA mode, no patches needed")

import gradio as gr
from shared.utils.audio_video import save_video
```

### 6.3 Google Drive integration — REAL implementation (not stubs)

Do not ship the earlier "always returns None / always disabled" placeholder functions. Implement real Google Drive upload via a **GCP service account** (the approach the user previously fixed for Shared-Drive uploads), reading credentials from environment variables that the notebook's Step 6 cell populates from Kaggle Secrets:

```python
GDRIVE_SA_JSON = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "")
GDRIVE_SHARED_DRIVE_ID = os.environ.get("GDRIVE_SHARED_DRIVE_ID", "")
GDRIVE_AVAILABLE = False
_gdrive_service = None

def _init_gdrive():
    """Initialize a Google Drive v3 API client from a service-account JSON
    string stored in the GDRIVE_SERVICE_ACCOUNT_JSON env var. Returns None
    (and leaves GDRIVE_AVAILABLE False) if not configured or on any error —
    this must NEVER crash the rest of the script."""
    global _gdrive_service, GDRIVE_AVAILABLE
    if not GDRIVE_SA_JSON:
        print("  [GDrive] No service account configured — backup disabled, bulk queue will keep local files.")
        return None
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        info = json.loads(GDRIVE_SA_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        GDRIVE_AVAILABLE = True
        print("  [GDrive] ✅ Service account authenticated")
        return service
    except Exception as e:
        print(f"  [GDrive] ⚠️ Initialization failed: {e} — backup disabled, bulk queue will keep local files.")
        return None

_gdrive_service = _init_gdrive()

def upload_video_to_gdrive(video_path, folder_id=None):
    """Upload a single file to Drive (or a Shared Drive if GDRIVE_SHARED_DRIVE_ID
    is set). Returns the file ID on success, None on any failure — callers must
    treat None as 'do not delete the local file'."""
    if not GDRIVE_AVAILABLE or _gdrive_service is None or not video_path or not os.path.exists(video_path):
        return None
    try:
        from googleapiclient.http import MediaFileUpload
        filename = os.path.basename(video_path)
        metadata = {"name": filename}
        if folder_id:
            metadata["parents"] = [folder_id]
        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        kwargs = {"body": metadata, "media_body": media, "fields": "id"}
        if GDRIVE_SHARED_DRIVE_ID:
            kwargs["supportsAllDrives"] = True
        file = _gdrive_service.files().create(**kwargs).execute()
        return file.get("id")
    except Exception as e:
        print(f"  [GDrive] ❌ Upload failed for {video_path}: {e}")
        return None

def _get_or_create_gdrive_folder(folder_name="LTX_Videos_Backup"):
    """Find or create a folder, scoped to the Shared Drive if configured."""
    if not GDRIVE_AVAILABLE:
        return None
    try:
        query = f"name='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"
        kwargs = {"q": query, "fields": "files(id, name)"}
        if GDRIVE_SHARED_DRIVE_ID:
            kwargs.update(
                corpora="drive",
                driveId=GDRIVE_SHARED_DRIVE_ID,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
        results = _gdrive_service.files().list(**kwargs).execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]
        metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
        if GDRIVE_SHARED_DRIVE_ID:
            metadata["parents"] = [GDRIVE_SHARED_DRIVE_ID]
        kwargs2 = {"body": metadata, "fields": "id"}
        if GDRIVE_SHARED_DRIVE_ID:
            kwargs2["supportsAllDrives"] = True
        folder = _gdrive_service.files().create(**kwargs2).execute()
        return folder.get("id")
    except Exception as e:
        print(f"  [GDrive] ⚠️ Folder lookup/create failed: {e}")
        return None

GDRIVE_FOLDER_ID = _get_or_create_gdrive_folder() if GDRIVE_AVAILABLE else None
```

Why a service account instead of `pydrive`/Colab-style OAuth: this notebook runs unattended on Kaggle (no interactive browser auth flow available), and the user specifically needs **Shared Drive** upload support (regular "My Drive" uploads via a service account silently fail with a storage-quota error on personal Drives — service accounts have no personal storage quota — so Shared Drive support via `supportsAllDrives=True` and a `driveId`/Shared-Drive-scoped folder query is mandatory, not optional, for this to actually work).

### 6.4 GGUF registration, config-loading patch, P100 audio-encoder wrapper, model loading

Keep the original's approach here — it's a legitimate, well-reasoned set of compatibility shims:
- `_register_gguf_handler()` — imports `shared.qtypes.gguf` so mmgp's `quant_router` knows how to read `.gguf` weight files.
- `_patch_ltx2_config_loading()` — monkey-patches `models.ltx2.ltx2._load_config_from_checkpoint` to read config from GGUF metadata first, falling back to a JSON file on disk if metadata read fails, instead of raising.
- `_AudioEncoderP100Wrapper` — wraps the audio encoder so that, under mmgp Profile 4's CPU/GPU offloading, the mel-spectrogram tensor is forced onto the same CUDA device + FP16 dtype as the encoder's weights before the forward pass, fixing a `cudaErrorNoKernelImageForDevice` that occurs specifically because mmgp moves the encoder to GPU but leaves a CPU-resident input tensor stale.
- After `family_handler.load_model(...)`, only when `_IS_SM60` is true, patch `CausalConv2d.forward` to run `pad`+`conv2d` on CPU in FP32 (sm_60 has no native BF16 kernels and PyTorch 2.4+ dropped sm_60 CUDA kernels for many ops entirely — running this one specific op on CPU is the targeted, narrow fix, not a blanket CPU fallback for everything).
- Apply `offload.profile(pipe, profile_no=4, quantizeTransformer=False, convertWeightsFloatTo=torch.float16, budgets={...})` with the same per-component VRAM budgets as the original (transformer 6000MB, text_encoder 1500MB, video_encoder 2000MB, video_decoder 3000MB, audio_encoder 1000MB, audio_decoder 1000MB, vocoder 500MB, spatial_upsampler 1500MB, vae 1000MB, default 1000MB) — these were empirically tuned for a 16GB T4 to land around a 9-minute total generation time and should be preserved as-is.

Use `MODEL_DTYPE = torch.float16` and `VAE_DTYPE = torch.float16` for both T4 and P100 (the original's reasoning that FP16 VAE decode is ~8x faster than FP32 on this hardware class is sound and applies to T4 too, not just P100).

### 6.5 Helper functions (keep from original, these are correct)
- `get_resolution(base_res_str, aspect_ratio_str)` — maps the four resolution presets and five aspect-ratio presets to a `(width, height)` pair, rounded down to the nearest multiple of 32 (required by the VAE's spatial downsampling factor).
- `get_vae_tile_size(height, width)` — computes a VAE tiling size from available VRAM and target resolution to avoid OOM on tall/wide outputs. NOTE: in the final script, **hardcode this to `256`** as the original script does in its actual call site (the function exists but the call site overrides it with a fixed value "to match the audio notebook and prevent OOM/slowdowns at higher res") — keep this hardcoded-256 behavior, it was a deliberate, tested choice, but keep the dynamic function too in case a future maintainer wants to re-enable it; just don't wire it back up automatically.
- `snap_to_ltx_frames(duration_sec, fps=24.0, max_frames=721)` — snaps to the nearest valid `8k+1` frame count the LTX distilled model requires, floor 49 frames (2s), cap 721 frames (30s). (This exists for potential audio-duration-driven generation; the dropdown-based duration selector in the UI doesn't need it directly, but keep the function available since the bulk CSV format allows future numeric-seconds duration values.)

### 6.6 `Video_Generation(...)` — the single-video generation function

Keep the original's structure and behavior exactly: builds `gen_kwargs`, calls `ltx2_model.generate(**gen_kwargs)`, handles the dict/tuple/bare-tensor result shapes defensively, saves video via `save_video(...)`, and if native audio was generated, muxes it into the MP4 via `ffmpeg` (re-encoding audio to AAC, copying video stream, `-shortest`). Preserve the two-stage progress callback (`cb`) that reports "Stage 1 (half-res)" vs "Stage 2 (full-res refine)" based on `pass_no`, and the `set_progress_status` callback for VAE-encode/decode/upsampling status labels — these give users real-time feedback during the ~9-minute generation and should not be simplified away.

**One required change from the original:** output filenames must use an incrementing counter (`ltx_video_0001.mp4`, `ltx_video_0002.mp4`, ...), not a Unix timestamp, per Failure #4 above. Implement via:
```python
OUTPUT_DIR = "/kaggle/working/outputs"
_output_counter = 0

def _get_next_output_path():
    global _output_counter
    _output_counter += 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"ltx_video_{_output_counter:04d}.mp4")
```
and use `out_path = _get_next_output_path()` inside `Video_Generation`, replacing the original's `timestamp = str(int(time.time())); out_path = os.path.join(OUTPUT_DIR, f"ltx_video_{timestamp}.mp4")`.

`progress` parameter default must be `gr.Progress()` (matching how Gradio actually invokes it when wired to a button click), not `None` — using `None` as the default and then calling `progress(0, desc=...)` unconditionally will crash with `TypeError: 'NoneType' object is not callable` the moment this function is called from anywhere other than a Gradio event with progress wired up (e.g. from the bulk runner, which the original code does call it from, with `progress=None` passed positionally in some call sites — audit every call site and make sure a valid no-op-capable progress object is always passed; the cleanest fix is the `gr.Progress()` default plus making the bulk runner accept its OWN `gr.Progress()` and forward a lightweight wrapper, see §6.7).

### 6.7 Bulk queue processing — ONE function only (Failure #4 fix)

Implement exactly one bulk-runner function, `run_bulk_queue_with_auto_delete`, combining the best of both prior versions:

```python
_bulk_stop_flag = threading.Event()

def _parse_bulk_csv(csv_text: str):
    """Parse the bulk CSV per the schema in spec §5. Returns a list of dicts."""
    jobs = []
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    for row in reader:
        row = {k.strip().lower(): v.strip() if isinstance(v, str) else v for k, v in row.items() if k}
        jobs.append({
            "prompt":       row.get("prompt", ""),
            "start_image":  row.get("start_image", row.get("start image", "")),
            "end_image":    row.get("end_image",   row.get("end image",   "")),
            "duration":     row.get("duration",    ""),
            "resolution":   row.get("resolution",  ""),
            "aspect_ratio": row.get("aspect_ratio", row.get("aspect ratio", "")),
            "seed":         int(row["seed"]) if row.get("seed", "").strip() not in ("", None) else -1,
            "guide_scale":  float(row["guide_scale"]) if row.get("guide_scale", row.get("guide scale", "")).strip() not in ("", None) else None,
            "steps":        int(row["steps"]) if row.get("steps", "").strip() not in ("", None) else None,
        })
    return jobs

def _extract_images_zip(zip_path, dest_dir):
    """Unzip the images archive, return {basename: full_path}."""
    os.makedirs(dest_dir, exist_ok=True)
    img_map = {}
    if zip_path and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    zf.extract(name, dest_dir)
                    img_map[os.path.basename(name)] = os.path.join(dest_dir, name)
    return img_map

def run_bulk_queue_with_auto_delete(csv_file, zip_file,
                                     default_duration, default_resolution, default_aspect,
                                     default_guide, default_steps,
                                     progress=gr.Progress()):
    """The ONE bulk-queue runner. For each row: resolve images, generate,
    upload to Drive if configured, delete locally ONLY if the upload
    succeeded (or if Drive isn't configured at all, in which case files
    are kept locally — never silently lose a video the user has no
    backup of)."""
    global _bulk_stop_flag
    _bulk_stop_flag.clear()

    if csv_file is None:
        return "❌ Please upload a CSV file."

    try:
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            jobs = _parse_bulk_csv(f.read())
    except Exception as e:
        return f"❌ CSV parse error: {e}"
    if not jobs:
        return "❌ No jobs found in CSV."

    img_dir = "/kaggle/working/bulk_images"
    img_map = _extract_images_zip(zip_file, img_dir) if zip_file else {}

    total, passed, failed = len(jobs), [], []
    log_lines = [f"🚀 Starting {total} job(s)…", ""]

    def _resolve(name):
        if not name:
            return None
        for candidate in (name, os.path.join("/kaggle/working", name), img_map.get(os.path.basename(name), "")):
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    for idx, job in enumerate(jobs):
        if _bulk_stop_flag.is_set():
            log_lines.append("🛑 Stopped by user.")
            break

        progress(idx / total, desc=f"Job {idx+1}/{total}: {job['prompt'][:50]}…")
        log_lines.append(f"[{idx+1}/{total}] {job['prompt'][:80]}")

        try:
            out_path, status = Video_Generation(
                prompt=job["prompt"],
                input_image_start=_resolve(job["start_image"]),
                input_image_end=_resolve(job["end_image"]),
                seed=job["seed"],
                duration_dropdown=job["duration"] or default_duration,
                resolution_dropdown=job["resolution"] or default_resolution,
                aspect_ratio_dropdown=job["aspect_ratio"] or default_aspect,
                guide_scale=job["guide_scale"] if job["guide_scale"] is not None else default_guide,
                num_steps=job["steps"] if job["steps"] is not None else default_steps,
                progress=progress,
            )
        except Exception as e:
            failed.append(job["prompt"])
            log_lines.append(f"  ❌ Exception: {e}")
            gc.collect(); torch.cuda.empty_cache()
            continue

        if not (out_path and os.path.exists(out_path)):
            failed.append(job["prompt"])
            log_lines.append(f"  ❌ {status}")
            gc.collect(); torch.cuda.empty_cache()
            continue

        passed.append(out_path)
        log_lines.append(f"  ✅ Generated: {os.path.basename(out_path)}")

        if GDRIVE_AVAILABLE:
            file_id = upload_video_to_gdrive(out_path, GDRIVE_FOLDER_ID)
            if file_id:
                log_lines.append(f"  ✅ Backed up to Google Drive ({file_id})")
                try:
                    os.remove(out_path)
                    log_lines.append("  🗑️  Auto-deleted from local storage (backup confirmed)")
                except Exception as e:
                    log_lines.append(f"  ⚠️  Backed up but could not delete local copy: {e}")
            else:
                log_lines.append("  ⚠️  Drive upload failed — keeping local copy for safety")
        else:
            log_lines.append("  ℹ️  Google Drive not configured — keeping local copy")

        gc.collect(); torch.cuda.empty_cache()

    progress(1.0, desc="✅ Batch complete!")
    log_lines.append("")
    log_lines.append(f"{'═'*50}")
    log_lines.append(f"✅ Completed: {len(passed)}  ❌ Failed: {len(failed)}  Total: {total}")
    log_lines.append(f"{'═'*50}")

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)

    return "\n".join(log_lines)

def stop_bulk():
    _bulk_stop_flag.set()
    return "🛑 Stop signal sent — current job will finish, then the queue halts."
```

Critical correctness points the previous implementation got wrong and you must get right:
- **Never delete a local video unless the Drive upload actually returned a real file ID.** The previous "auto-delete" version had a code path where `GDRIVE_AVAILABLE` was always `False` (because the Drive functions were stubs), yet it still printed `"🗑️ Auto-deleted from local storage"` style messages in places — audit your final logic so deletion is strictly gated on upload success, never on "Drive was configured" alone.
- `guide_scale`/`steps` per-row overrides must distinguish "column present but empty → use default" from "explicit `0`" — use `None` as the sentinel (not falsy-string-or-zero checks) so a user who explicitly sets `steps=0` in a CSV (even though that's an unusual choice) isn't silently overridden by the default. (`0 or default_steps` would incorrectly replace an explicit `0`; the `is not None` check above avoids this.)
- One single `progress` object must be threaded through from the Gradio button click → `run_bulk_queue_with_auto_delete` → each `Video_Generation` call, so progress bars update smoothly across the whole batch rather than resetting to 0% on every row.

### 6.8 Gradio UI — built correctly from the start, no post-hoc patching

Write `gr.Blocks(...)` with `gr.Tabs()` and the two `gr.TabItem(...)` blocks (`"🎬 Single Video"` and `"📦 Bulk Queue"`) as the actual, final, correctly-indented structure in this one `%%writefile` cell — per Failure #3, do not write a flat single-tab UI first and patch it later.

Single Video tab — keep the original's well-thought-out layout: prompt textbox, collapsible "Image to Video (Optional)" accordion with start/end frame image inputs and explanatory markdown about T2V vs I2V vs first+last-frame modes, seed number input, duration/resolution/aspect-ratio dropdowns, guide-scale and steps sliders, Generate/Stop/Clear buttons, video output, status textbox, and a collapsible "Output Manager" accordion with refresh/backup/delete controls wired to the real Drive functions from §6.3 (not stubs).

Bulk Queue tab — CSV + ZIP file uploads, default-value dropdowns/sliders matching the single-video tab's options (these serve as fallbacks for CSV rows that omit a column), Run/Stop buttons, and a single log textbox (15 lines) showing `run_bulk_queue_with_auto_delete`'s returned log string. Include the example-CSV markdown block showing the exact 9-column header and 1-2 example rows.

Keep the CSS block, color scheme, and branding elements from the original (the purple/blue gradient header, social buttons, footer) — cosmetic details aren't the issue here, structural correctness is.

Launch with:
```python
demo.queue()
demo.launch(share=True, inline=False, debug=True, show_error=True, max_threads=1, ssr_mode=False)
```

## 7. VALIDATION CHECKLIST — VERIFY BEFORE CONSIDERING THIS DONE

After Trae generates the notebook, walk through this checklist explicitly and fix anything that fails:

1. [ ] There is exactly one `.ipynb` file in the repo. No second "simplified" notebook exists anywhere.
2. [ ] `grep` the notebook source for any cell that does `os.path.exists('/kaggle/input/datasets/...')` — there must be zero matches. The script must never depend on an attached Kaggle Dataset for its own source code.
3. [ ] The torch-pin assertion cell exists and runs immediately after the dependency-install cell, before any model loading.
4. [ ] `run_ltx_t2v.py` is written by exactly one `%%writefile` cell. Search for any cell using `.replace(`, `re.sub(`, or a Python heredoc that opens and rewrites `run_ltx_t2v.py` after it was first written — there must be zero such cells.
5. [ ] `grep` the generated `run_ltx_t2v.py` for `def run_bulk_queue` — it must appear exactly once (as `run_bulk_queue_with_auto_delete`), not twice under two different names.
6. [ ] `grep` for `OUTPUT_DIR = ` — must appear exactly once.
7. [ ] `grep` for `def list_outputs` — must appear exactly once.
8. [ ] Confirm `upload_video_to_gdrive` actually calls the Drive API (not a function that unconditionally `return None`).
9. [ ] Confirm bulk auto-delete is strictly gated on a non-`None` upload result, never on `GDRIVE_AVAILABLE` alone.
10. [ ] Confirm `Video_Generation`'s `progress` parameter defaults to `gr.Progress()`, not `None`, and that every call site (single-video button click, bulk runner) passes a valid progress object.
11. [ ] Confirm `_parse_bulk_csv` reads with `encoding='utf-8-sig'` and accepts both `start_image`/`start image` and `end_image`/`end image` header variants.
12. [ ] Confirm output filenames use the `ltx_video_NNNN.mp4` incrementing-counter scheme, not Unix timestamps.
13. [ ] Confirm the notebook can be read top-to-bottom and "Run All" works on a completely fresh Kaggle kernel with GPU T4 x1, zero attached datasets, and (optionally) the two Kaggle Secrets for Drive backup either present or absent — both cases must complete without a crash.
14. [ ] `sample_jobs/jobs_template.csv` and `sample_jobs/jobs_example.csv` exist, are valid CSV, and use the exact 9-column header from §5.

Build all of this now. Where any instruction above is ambiguous, prefer the explicit, defensive, "never silently lose a user's GPU-minutes-expensive video" behavior over a shorter or cleverer implementation.
