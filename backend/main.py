from base64 import b64encode
from io import BytesIO
import os
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

app = FastAPI(
    title="AcneClear Photo Enhancement API",
    version="1.0.0",
    description="In-memory photo enhancement using pretrained GFPGAN v1.3.",
)

web_origins = os.getenv("ACNECLEAR_WEB_ORIGIN", "").strip()
allowed_origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8081", "http://127.0.0.1:8081",
]
if web_origins:
    allowed_origins.extend(origin.strip() for origin in web_origins.split(",") if origin.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_DIR = Path(os.getenv("ACNECLEAR_MODEL_DIR", str(Path(__file__).resolve().parent / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_PATH = MODEL_DIR / "GFPGANv1.3.pth"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
_GFPGAN_RESTORER = None


def get_restorer():
    global _GFPGAN_RESTORER
    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from gfpgan import GFPGANer
    if not WEIGHT_PATH.exists():
        urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
    _GFPGAN_RESTORER = GFPGANer(
        model_path=str(WEIGHT_PATH), upscale=2, arch="clean",
        channel_multiplier=2, bg_upsampler=None,
    )
    return _GFPGAN_RESTORER


def soften_texture(image_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=50, sigmaSpace=50)
    blended = cv2.addWeighted(bgr, 0.78, smooth, 0.22, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


@app.get("/")
def root():
    return {"name": "AcneClear Photo Enhancement API", "status": "running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "acneclear-photo-enhancement",
        "model": "GFPGAN v1.3",
        "model_loaded": _GFPGAN_RESTORER is not None,
    }


@app.post("/api/enhance")
def enhance_photo(file: UploadFile = File(...)):
    """Enhance one face photo and return a PNG as base64. Uploaded files are not saved."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG or WEBP image.")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as source:
            source.verify()
        with Image.open(BytesIO(contents)) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")

    rgb = np.asarray(original)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        restorer = get_restorer()
        _, _, restored_bgr = restorer.enhance(
            bgr, has_aligned=False, only_center_face=False, paste_back=True, weight=0.6
        )
        if restored_bgr is None:
            raise RuntimeError("No face could be restored.")
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(soften_texture(restored_rgb))
        note = "GFPGAN facial restoration and gentle texture smoothing are complete."
    except AssertionError:
        result = Image.fromarray(soften_texture(rgb))
        note = "A face could not be aligned, so a gentle smoothing-only preview was created."
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Photo enhancement failed ({type(exc).__name__}). Try a clear, front-facing photo.",
        )

    output = BytesIO()
    result.save(output, format="PNG", optimize=True)
    return {
        "filename": file.filename,
        "content_type": "image/png",
        "image_base64": b64encode(output.getvalue()).decode("ascii"),
        "note": note,
    }
