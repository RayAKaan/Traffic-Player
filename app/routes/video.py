# app/routes/video.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.database import VideoTask, get_session
from src.model.predict import process_video_with_model
from sqlmodel import select
import os, shutil, uuid, threading
import time

router = APIRouter()

RAW_DIR = "SmarTSignalAI/data/raw"
PROCESSED_DIR = "SmarTSignalAI/data/processed"

# ---------------- Video Processing Endpoint ----------------
@router.post("/process_video/")
async def process_video(file: UploadFile = File(...), enhanced: str = Form("false")):
    uid = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    raw_path = os.path.join(RAW_DIR, f"{uid}.{ext}")

    os.makedirs(RAW_DIR, exist_ok=True)

    # Save uploaded file
    with open(raw_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Save task in DB as queued
    with get_session() as session:
        task = VideoTask(id=task_id, filename=file.filename, status="queued", progress=0)
        session.add(task)
        session.commit()

    # ---------------- Background Processing ----------------
    progress_dict = {task_id: {"progress": 0}}

    def background_task():
        try:
            # Mark task as processing
            with get_session() as session:
                task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
                if task:
                    task.status = "processing"
                    session.add(task)
                    session.commit()

            # Process video
            processed_path, stats = process_video_with_model(
                input_path=raw_path,
                output_dir=PROCESSED_DIR,
                progress_store=progress_dict,
                task_id=task_id,
                enhanced=enhanced.lower() == "true"
            )

            # Final DB update: completed
            with get_session() as session:
                task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
                if task:
                    task.status = "completed"
                    task.progress = 100
                    task.processed_path = processed_path.replace("SmarTSignalAI/data/", "")
                    task.stats = stats
                    session.add(task)
                    session.commit()

        except Exception as e:
            print(f"[ERROR] Video task {task_id} failed: {e}")
            with get_session() as session:
                task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
                if task:
                    task.status = "failed"
                    session.add(task)
                    session.commit()

    # ---------------- Progress Updater ----------------
    def progress_updater():
        while True:
            time.sleep(1)  # update every 1 second
            current_progress = progress_dict.get(task_id, {}).get("progress", 0)
            with get_session() as session:
                task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
                if task and task.status == "processing":
                    task.progress = current_progress
                    session.add(task)
                    session.commit()
            if current_progress >= 100:
                break

    # Start background threads
    threading.Thread(target=background_task, daemon=True).start()
    threading.Thread(target=progress_updater, daemon=True).start()

    return JSONResponse({"task_id": task_id})


# ---------------- Task Status Endpoint ----------------
@router.get("/status/{task_id}")
async def get_status(task_id: str):
    with get_session() as session:
        task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
        if not task:
            raise HTTPException(404, "Task not found")
        return {
            "status": task.status,
            "progress": task.progress,
            "processed_path": task.processed_path,
            "stats": task.stats
        }
