"""Cell 1: Setup and environment verification."""


def run(context: dict) -> dict:
    import torch
    config = context.get("config")
    log = context.get("log")
    fs = context.get("fs")
    log.info("Cell 1: Setup started")
    log.info("PyTorch version: %s", torch.__version__)
    log.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    fs.write_project_config()
    log.info("Cell 1: Setup completed")
    return context
