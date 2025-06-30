# app/database.py

from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import os
import json

# Define the database URL and location
db_path = os.path.join("SmarTSignalAI", "data", "tasks.db")
DATABASE_URL = f"sqlite:///{db_path}"

# Initialize SQLModel engine
engine = create_engine(DATABASE_URL, echo=False)

# Define the Task table using SQLModel
class VideoTask(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    status: str = "queued"
    progress: int = 0
    processed_path: Optional[str] = None
    stats_json: Optional[str] = Field(default="{}")  # Store stats as JSON string

    # Optional: Convenience method to get dict from JSON string
    @property
    def stats(self) -> dict:
        return json.loads(self.stats_json or "{}")

    @stats.setter
    def stats(self, value: dict):
        self.stats_json = json.dumps(value)

# Create the tasks table on startup
def create_db_and_tables():
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    SQLModel.metadata.create_all(engine)

# Dependency for retrieving a session
def get_session():
    return Session(engine)
