from pydantic import BaseModel

class ProcessedVideoResponse(BaseModel):
    processed_path: str
