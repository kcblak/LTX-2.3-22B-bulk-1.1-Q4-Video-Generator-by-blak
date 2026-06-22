try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="LTX Notebook Monitor")
    _templates_dir = Path("templates")
    _templates_dir.mkdir(exist_ok=True)
    templates = Jinja2Templates(directory=str(_templates_dir))

    def _get_context():
        from core.config import Config
        from core.fs_manager import FSManager
        from core.logger import Logger
        from core.job_queue import JobQueue
        cfg = Config({
            "project_name": "MyProject",
            "mode": "headless",
            "device": "cpu",
            "batch_size": 1,
            "max_retries": 3,
            "parallel_jobs": 1,
            "output_format": "mp4",
            "resolution_default": "720p",
            "logging_level": "info",
        })
        fs = FSManager(cfg)
        logger = Logger(fs.logs_dir, cfg.logging_level)
        queue = JobQueue(fs, logger)
        return cfg, fs, logger, queue

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        cfg, fs, logger, queue = _get_context()
        status_path = fs.status_csv_path()
        rows = []
        if status_path.exists():
            import csv
            with status_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        return templates.TemplateResponse("index.html", {
            "request": request, "rows": rows,
            "summary": fs.summary(),
        })

    @app.get("/api/jobs")
    async def api_jobs():
        cfg, fs, logger, queue = _get_context()
        status_path = fs.status_csv_path()
        if not status_path.exists():
            return JSONResponse({"jobs": []})
        import csv
        with status_path.open("r", encoding="utf-8") as f:
            return JSONResponse({"jobs": list(csv.DictReader(f))})

    @app.get("/api/logs/{job_id}")
    async def api_job_log(job_id: str):
        cfg, fs, logger, queue = _get_context()
        path = fs.job_log_path(job_id)
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({"log": path.read_text(encoding="utf-8")})


def run_dashboard(host: str = "0.0.0.0", port: int = 8000):
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi and uvicorn are required for the dashboard")
    import uvicorn
    uvicorn.run(app, host=host, port=port)
