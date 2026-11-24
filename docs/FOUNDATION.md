# AcneClear AI foundation

The current project is intentionally focused on one capability: **photo enhancement**.

## Architecture

- Root app.py: Streamlit photo-enhancement interface.
- backend/: FastAPI service with the POST /api/enhance endpoint.
- frontend/website/: React + Vite website.
- mobile/: Expo React Native photo-enhancement client.
- models/: locally downloaded GFPGAN checkpoint; excluded from Git.

## Enhancement pipeline

1. The client sends one JPG, PNG or WEBP photo to the API.
2. The backend validates the file and reads it into memory.
3. Pretrained GFPGAN v1.3 restores facial details.
4. A restrained OpenCV bilateral-smoothing blend is applied.
5. The backend returns a PNG preview as base64 for the client to display or download.

The uploaded image is not written to disk. The model checkpoint is downloaded separately and stored in the ignored models directory.

## Run the services

Backend:

~~~bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
~~~

Website:

~~~bash
cd frontend/website
npm install
echo "VITE_API_URL=http://127.0.0.1:8000" > .env
npm run dev
~~~

## API

- GET /api/health
- POST /api/enhance with multipart form field file

The endpoint accepts image files up to 10 MB.

## Limitations

GFPGAN is a general face-restoration model, not a dedicated acne-removal model. It does not detect acne type, diagnose skin disease or guarantee removal of individual marks. Image quality, lighting and face angle affect the result. The application is for photo enhancement only and is not a medical service.
