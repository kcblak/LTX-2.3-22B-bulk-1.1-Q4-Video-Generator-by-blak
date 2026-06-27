# LTX-2.3 22B Notebook System

A self-hosted, Kaggle-like notebook environment for bulk image-to-video and prompt-driven animation generation.

## Kaggle Workflow

1. Add dataset containing `jobs.csv` and optional `images.zip` to your Kaggle notebook
2. Set secret `GDRIVE_SERVICE_ACCOUNT_JSON` (paste full JSON string)
3. Run: `python main.py --init --project MyProject && python main.py --run --headless`
4. System auto-loads from `/kaggle/input/`, writes to `/kaggle/working/MyDrive/LTX_PROJECTS/`

## Features

- CSV-driven batch job processing
- Headless execution (no UI required)
- Google Drive integration
- Automatic checkpointing and resume
- Structured logging per job
- Modular pipeline with model plugins
- Fault tolerance with retry and OOM recovery

## Quick Start

1. Copy `config.example.yaml` to `config.yaml` and edit as needed.
2. Run `python main.py --init --project MyProject`
3. Place your `jobs.csv` in `MyDrive/LTX_PROJECTS/MyProject/input/jobs.csv`
4. Run `python main.py --run --headless`

## Commands

| Command | Description |
|---------|-------------|
| `python main.py init --project <name>` | Initialize project directory |
| `python main.py run --headless` | Run full pipeline |
| `python main.py resume` | Resume interrupted run |
| `python main.py status` | Show job status |
| `python main.py export-logs` | Archive logs |

## Directory Structure

```
MyDrive/LTX_PROJECTS/Project_Name/
├── input/
│   ├── jobs.csv
│   ├── images/
│   ├── zips/
│   └── extracted/
├── output/
│   ├── videos/
│   ├── frames/
│   └── thumbnails/
├── logs/
├── checkpoints/
├── cache/
└── config/
```

## CSV Format (DO NOT MODIFY)

```
prompt,start_image,end_image,duration,resolution,aspect_ratio,seed,guide_scale,steps
```

## License

MIT
