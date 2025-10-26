# app/tasks/video_tasks.py
import os
import traceback
from src.model.predict import process_video_with_model
from app.database import VideoTask, get_session
from sqlmodel import select
from app.celery_app import celery

# Define directories
RAW_DIR = "SmarTSignalAI/data/raw"
PROCESSED_DIR = "SmarTSignalAI/data/processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def process_video_task(self, task_id: str, raw_path: str, enhanced: bool = False):
    """
    Celery task to process a video in the background.
    It updates progress live via predict.py and handles retries safely.
    """

    print(f"[CELERY] Starting background task for video: {task_id}")

    try:
        # --- Step 1: Mark task as 'processing' in DB ---
        with get_session() as session:
            task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
            if task:
                task.status = "processing"
                task.progress = 0
                session.add(task)
                session.commit()

        # --- Step 2: Process video (this function updates DB progress internally) ---
        processed_path, stats = process_video_with_model(
            input_path=raw_path,
            output_dir=PROCESSED_DIR,
            task_id=task_id,
            enhanced=enhanced,
        )

        relative_path = processed_path.replace("SmarTSignalAI/data/", "")

        # --- Step 3: Mark task as completed ---
        with get_session() as session:
            task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
            if task:
                task.status = "completed"
                task.progress = 100
                task.processed_path = relative_path
                task.stats = stats
                session.add(task)
                session.commit()

        print(f"[CELERY] ✅ Completed task {task_id}")
        return {"processed_path": relative_path, "stats": stats}

    except Exception as e:
        print(f"[CELERY] ❌ Task {task_id} failed: {e}")
        traceback.print_exc()

        # --- Step 4: Update DB to failed ---
        with get_session() as session:
            task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
            if task:
                task.status = "failed"
                task.progress = 0
                session.add(task)
                session.commit()

        # Retry up to 3 times
        raise self.retry(exc=e)
