# 🚀 LTX-2.3 22B HEADLESS BATCH MODE - Complete Kaggle Guide

## Prerequisites
1. Add dataset: `kcblak/ltx-batch` (contains `jobs.csv` + `images.zip`)
2. Set Kaggle secret: `GDRIVE_SERVICE_ACCOUNT_JSON` (full JSON string)

---

## Step 1: Environment Setup
```python
import os, psutil, subprocess
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
!df -h /kaggle/working /kaggle/tmp
```

## Step 2: Clone Wan2GP & Install PyTorch
```python
if not os.path.isdir("Wan2GP"):
    !git clone --depth 1 https://github.com/DeepBeepMeep/Wan2GP.git
    
!pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
**Restart kernel** (Kernel → Restart Kernel) before continuing.

## Step 3: Install Core Dependencies (FIXED - no conflicts)
```python
DEPS = [
    "diffusers>=0.25.0", "transformers>=4.40.0", "accelerate>=0.30.0",
    "safetensors>=0.4.0", "Pillow>=10.0.0", "opencv-python>=4.9.0",
    "sentencepiece>=0.2.0", "peft>=0.8.0", "huggingface-hub>=0.20.0",
    "ffmpeg-python>=0.2.0",  # REQUIRED for save_video
]
for dep in DEPS:
    !pip install --timeout 120 -q $dep

!pip install --timeout 120 -q mmgp gradio gguf soundfile google-api-python-client
```

## Step 4: Load Google Drive Secret (REQUIRED for auto-upload)
```python
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    sa_json = user_secrets.get_secret("GDRIVE_SERVICE_ACCOUNT_JSON") or ""
    if sa_json:
        os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"] = sa_json
        print("✅ Google Drive secret loaded")
    else:
        print("⚠️ Set GDRIVE_SERVICE_ACCOUNT_JSON secret in Kaggle")
except Exception as e:
    print(f"Secret load failed: {e}")
```

## Step 5: Clone Repo & Initialize
```python
%cd /kaggle/working
!git clone https://github.com/kcblak/LTX-2.3-22B-bulk-1.1-Q4-Video-Generator-by-blak
%cd LTX-2.3-22B-bulk-1.1-Q4-Video-Generator-by-blak

!python main.py --init --project MyLTXProject
```

## Step 6: Run Headless Pipeline
```python
!python main.py --run --headless
```

This will:
- Read `/kaggle/input/ltx-batch/jobs.csv`
- Extract `/kaggle/input/ltx-batch/images.zip`
- Generate videos with auto-upload to Google Drive
- Log to `/kaggle/working/MyDrive/LTX_PROJECTS/MyLTXProject/logs/`

---

## Commands Reference
| Command | Description |
|---------|-------------|
| `python main.py --init --project NAME` | Create project directories |
| `python main.py --run --headless` | Run full batch pipeline |
| `python main.py --resume` | Resume interrupted run |
| `python main.py --status` | Show job status |
| `python main.py --export-logs` | Archive logs to zip |

---

## Output Locations
- **Videos**: `MyDrive/LTX_PROJECTS/NAME/output/videos/job_*.mp4`
- **Logs**: `MyDrive/LTX_PROJECTS/NAME/logs/job_*.log` + `system.log`
- **Status**: `MyDrive/LTX_PROJECTS/NAME/checkpoints/job_status.csv`

---

## CSV Format (DO NOT MODIFY)
```
prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guide_scale,steps
```