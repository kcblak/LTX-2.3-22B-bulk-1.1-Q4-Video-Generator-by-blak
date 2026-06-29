# 🚀 HEADLESS BATCH MODE - Step by Step Guide

This notebook now supports FULLY HEADLESS operation for bulk video generation.

---

## Prerequisites

1. **Add Dataset**: Search and add `kcblak/ltx-batch` in Kaggle (contains `jobs.csv` + `images.zip`)
2. **Set Secret**: Add `GDRIVE_SERVICE_ACCOUNT_JSON` to Kaggle Secrets (full JSON string)

---

## Dependency Fix (Kaggle-specific)

The `Wan2GP/requirements.txt` has conflicting versions. Run this BEFORE Step 3:

```python
!pip install --timeout 120 diffusers transformers accelerate safetensors Pillow opencv-python sentencepiece peft huggingface-hub ffmpeg-python
!pip install --timeout 120 mmgp gradio gguf soundfile google-api-python-client
```

---

## Load Google Drive Secret (Before running headless)

```python
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    sa_json = user_secrets.get_secret("GDRIVE_SERVICE_ACCOUNT_JSON") or ""
    import os
    os.environ["GDRIVE_SERVICE_ACCOUNT_JSON"] = sa_json
    print("✅ Google Drive secret loaded")
except Exception as e:
    print(f"ℹ️ Secrets unavailable or not configured: {e}")
```

---

## Execution Steps

### Step 1: Initialize Project
```bash
!python main.py --init --project MyProject
```

### Step 2: Run Headless Pipeline
```bash
!python main.py --run --headless
```

The system will:
- Read `jobs.csv` from `/kaggle/input/ltx-batch/jobs.csv`
- Extract images from `/kaggle/input/ltx-batch/images.zip`
- Generate videos one by one
- Auto-upload each to Google Drive
- Log progress to `/kaggle/working/MyDrive/LTX_PROJECTS/MyProject/logs/`

### Commands Reference

| Command | Purpose |
|---------|---------|
| `python main.py --init --project NAME` | Create project directories |
| `python main.py --run --headless` | Run full pipeline, auto-upload to Drive |
| `python main.py --resume` | Resume interrupted run |
| `python main.py --status` | Show job status from CSV |
| `python main.py --export-logs` | Archive logs to zip |

---

## Output Locations

- **Videos**: `MyDrive/LTX_PROJECTS/NAME/output/videos/job_*.mp4`
- **Logs**: `MyDrive/LTX_PROJECTS/NAME/logs/system.log` and `logs/job_*.log`
- **Status**: `MyDrive/LTX_PROJECTS/NAME/checkpoints/job_status.csv`

---

## CSV Format (DO NOT CHANGE)

```
prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guide_scale,steps
```

Example:
```csv
"A majestic eagle",,,5 Seconds (121 frames),720p,16:9 Landscape,-1,3.0,8
"City timelapse",city.jpg,,3 Seconds (73 frames),540p,1:1 Square,42,2.5,6
```