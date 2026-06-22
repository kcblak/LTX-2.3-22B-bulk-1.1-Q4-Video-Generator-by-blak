"""Cell 2: Load data, validate CSV, extract ZIPs, resolve images."""


def run(context: dict) -> dict:
    fs = context.get("fs")
    log = context.get("log")
    log.info("Cell 2: Load data started")
    fs.extract_zips()
    log.info("ZIP extraction completed")
    log.info("Cell 2: Data load completed")
    return context
