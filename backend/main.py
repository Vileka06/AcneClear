from io import BytesIO
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="AcneClear AI API",
    version="0.1.0",
    description="Backend foundation for the AcneClear AI website and mobile app.",
)

# Development origins. Set ACNECLEAR_WEB_ORIGIN for your deployed website later.
web_origin = os.getenv("ACNECLEAR_WEB_ORIGIN", "").strip()
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]
if web_origin:
    allowed_origins.extend(origin.strip() for origin in web_origin.split(",") if origin.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def root():
    return {"name": "AcneClear AI API", "version": "0.1.0", "status": "running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "acneclear-api",
        "models": {
            "dcgan": "not_configured",
            "gfpgan": "not_configured",
            "skin_classifier": "not_configured",
        },
    }


@app.post("/api/uploads/validate")
async def validate_upload(file: UploadFile = File(...)):
    """Validate an image upload without storing it or running a model."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
        image = Image.open(BytesIO(contents))
        width, height = image.size
        image_format = image.format
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")

    return {
        "valid": True,
        "filename": file.filename,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(contents),
        "message": "Image validated. Skin prediction and enhancement are not connected yet.",
    }
