"""
FastAPI web app for SortVision.

Serves a single-page frontend with both file upload and webcam capture,
an inference endpoint, and a Grad-CAM explanation endpoint.

Run with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 in a browser.
"""

import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config
from src.gradcam import compute_gradcam_heatmap, overlay_heatmap

app = FastAPI(title="SortVision")

# Allows a separately hosted frontend (e.g. on Vercel) to call this API.
# Tighten allow_origins to your actual frontend domain before going live;
# "*" is fine for local development and quick demos only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(config.APP_MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No trained model found at {config.APP_MODEL_PATH}. "
                    "Train one first with: python -m src.train"
                ),
            )
        _model = tf.keras.models.load_model(config.APP_MODEL_PATH)
    return _model


def check_image_plausibility(img: Image.Image) -> str | None:
    """Same spirit as the breast cancer project's check: a lightweight
    heuristic, not a learned detector, flagging images unlikely to be a
    real photo of an object at all (for example, a near-solid color
    image, which a webcam can produce if the lens is covered or pointed
    at a blank wall).
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    std_dev = float(arr.std())

    if std_dev < 0.03:
        return (
            "This image has very little visual detail. Make sure the "
            "object is in frame and well lit before relying on this "
            "result."
        )
    return None


def _read_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image file.")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    model = get_model()
    raw = await file.read()
    img = _read_image(raw)

    warning = check_image_plausibility(img)

    resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    ranked_indices = np.argsort(probs)[::-1]

    results = [
        {
            "class_name": config.CLASS_NAMES[i],
            "display_name": config.DISPLAY_NAMES.get(config.CLASS_NAMES[i], config.CLASS_NAMES[i]),
            "probability": float(probs[i]),
        }
        for i in ranked_indices
    ]

    return JSONResponse(
        {
            "top": results[0],
            "all": results,
            "warning": warning,
        }
    )


@app.post("/api/explain")
async def explain(file: UploadFile = File(...)):
    """Runs the same prediction, plus returns a Grad-CAM heatmap overlay
    as a PNG image showing which part of the image drove the prediction.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    model = get_model()
    raw = await file.read()
    img = _read_image(raw)

    resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    heatmap, class_index = compute_gradcam_heatmap(model, arr)
    overlay = overlay_heatmap(resized, heatmap)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/health")
def health():
    return {
        "model_available": os.path.exists(config.APP_MODEL_PATH),
        "model_path": config.APP_MODEL_PATH,
        "classes": config.CLASS_NAMES,
    }