import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pathlib import Path

from app.routes.video import router as video_router
from app.database import create_db_and_tables

app = FastAPI(
    title="SmarTSignalAI Backend",
    description="YOLOv8-powered traffic detection and signal management system.",
    version="1.0.0"
)

# ------------------ Security & CORS ------------------
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
VIDEO_DATA_DIR = PROJECT_ROOT / "SmarTSignalAI" / "data"

# ------------------ Serve video files ------------------
class CustomStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".mp4"):
            response.headers["Content-Type"] = "video/mp4"
        return response

app.mount("/data", CustomStaticFiles(directory=VIDEO_DATA_DIR), name="data")

# ✅ Only serve frontend in production
if os.getenv("ENV", "dev") == "prod" and FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    print("[DEV MODE] Frontend not served by backend. Run `npm run dev` separately.")

# ------------------ API routers ------------------
app.include_router(video_router, prefix="/api/video", tags=["Video Processing"])

# ------------------ Startup ------------------
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
