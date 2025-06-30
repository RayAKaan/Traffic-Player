# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.routes.video import router as video_router  # Video upload + status routes
from app.database import create_db_and_tables

app = FastAPI(
    title="SmarTSignalAI Backend",
    description="YOLOv8-powered traffic detection and signal management system.",
    version="1.0.0"
)

# ✅ TrustedHost (optional for security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Replace with ["localhost", "127.0.0.1"] or domain in production
)

# ✅ CORS (Frontend Integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve processed video files as static
class CustomStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".mp4"):
            response.headers["Content-Type"] = "video/mp4"
        return response

app.mount("/data", CustomStaticFiles(directory="SmarTSignalAI/data"), name="data")

# ✅ Register API routers
app.include_router(video_router, prefix="/api/video", tags=["Video Processing"])

# ✅ Initialize database on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# ✅ Serve frontend files
@app.get("/", include_in_schema=False)
async def serve_frontend(request):
    return FileResponse("SmarTSignalAI/frontend/dist/index.html")