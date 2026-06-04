#!/bin/bash

echo "Running in $MODE mode"

if [ "$MODE" == "train" ]; then
    echo "Starting training..."
    exec python3 -u -m src.cli

elif [ "$MODE" == "serve" ]; then
    echo "Serving FastAPI model..."
    uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload

elif [ "$MODE" == "mlflow" ]; then
    echo "Starting MLflow UI..."
    mlflow ui --backend-store-uri /app/experiments/mlruns --host 0.0.0.0 --port 5000

else
    echo "Invalid mode: $MODE"
    exit 1
fi