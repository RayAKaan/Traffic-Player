# app/celery_app.py
from celery import Celery

celery = Celery(
    "smartsignalai",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

celery.conf.task_track_started = True
celery.conf.task_serializer = "json"
celery.conf.result_serializer = "json"
celery.conf.accept_content = ["json"]
celery.conf.worker_concurrency = 2  # optional: limit concurrent tasks
