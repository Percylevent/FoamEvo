from pydantic import BaseModel


class RunOut(BaseModel):
    job_id: str | None
    status: str  # "submitted" | "completed" | "failed"
