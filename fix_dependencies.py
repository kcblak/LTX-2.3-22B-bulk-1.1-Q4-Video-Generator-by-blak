# Dependency fix for Kaggle notebooks
# Run this in a cell BEFORE Step 3 to avoid conflicts

import subprocess
import sys

# Core packages without conflicting versions
CORE_DEPS = [
    "diffusers>=0.25.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "safetensors>=0.4.0",
    "Pillow>=10.0.0",
    "opencv-python>=4.9.0",
    "sentencepiece>=0.2.0",
    "peft>=0.8.0",
    "huggingface-hub>=0.20.0",
    "ffmpeg-python>=0.2.0",  # Required for Wan2GP save_video
]

print("Installing core dependencies individually...")
for dep in CORE_DEPS:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--timeout", "120", dep], check=True, capture_output=True)
        print(f"  ✓ {dep}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ {dep} - may already be installed")

print("\nInstalling mmgp, gradio, gguf, soundfile...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "--timeout", "120",
    "mmgp", "gradio", "gguf", "soundfile", "google-api-python-client",
], check=False)

print("\nSkip the Wan2GP/requirements.txt install - using pre-installed packages")