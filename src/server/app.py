from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.monitoring.metrics import metrics_router, observe_request
from src.server.inference import InferenceEngine

app = FastAPI(title="ML Inference Service", version="0.1.0")
app.include_router(metrics_router)

engine = InferenceEngine()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=1024)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(payload: GenerateRequest) -> dict:
    with observe_request():
        result = await engine.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
        )
    return result
