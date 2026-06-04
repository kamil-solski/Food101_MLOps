from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
import time

from inference_api.inference.loader import resolve_and_load
from inference_api.inference.predictor import run_inference_ort
from inference_api.utils.postprocessing import topk_indices, map_indices_to_labels

#TODO: add more elegant and complete way to shutdown FastAPI 

app = FastAPI(title="Food101 Inference API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION, META = resolve_and_load()

@app.get("/health")
def health():
    return {"status": "ok", "model_name": META["model_name"], "model_version": META["version"], "alias": META["alias"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    img_bytes = await file.read()
    t0 = time.time()
    
    probs = run_inference_ort(SESSION, META, img_bytes)
    idx = topk_indices(probs, int(META.get("topk", 5)))
    preds = map_indices_to_labels(idx, probs, META.get("classes"))
    
    latency_ms = round((time.time() - t0) * 1000.0, 2)

    resp = JSONResponse({
        "request_id": str(uuid4()),
        "served_alias": META["alias"],
        "model_version": META["version"],
        "predictions": preds,
        "latency_ms": latency_ms,
    })
    resp.headers["X-Model-Alias"] = META["alias"]
    resp.headers["X-Model-Version"] = META["version"]
    return resp

# To run locally outside docker container (while being inside project root folder):
# python -m uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload
# curl -s http://localhost:8000/health | jq
# curl -s -X POST http://localhost:8000/predict -F "file=@Data/<file_name>.png" | jq
# because inference docker is mounted to host over 8000 port we can also run above functions from host even for container runs