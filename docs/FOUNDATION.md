# AcneClear AI foundation

This branch adds a separate FastAPI backend, a responsive React website and an Expo React Native mobile starter. The existing Streamlit app remains unchanged.

## Architecture

- `app.py` at repository root: existing Streamlit application. Keep working while the new clients are developed.
- `backend/`: FastAPI API shared by website and mobile.
- `frontend/website/`: React + Vite website.
- `mobile/`: Expo React Native mobile app.
- `models/`: reserved for model adapters and checkpoints; do not commit large weights.

## Run the backend

Python 3.11 is recommended.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check http://localhost:8000/api/health and API docs at http://localhost:8000/docs.

## Run the website

In a second terminal:

```bash
cd frontend/website
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173.

## Run the mobile app

In a third terminal:

```bash
cd mobile
npm install
npx expo start
```

For a physical phone, change `API_URL` in `mobile/App.js` from localhost to your computer's LAN IP address (same Wi-Fi network). Android emulator commonly uses `http://10.0.2.2:8000`; iOS simulator can use `http://localhost:8000`.

## Current API scope

- `GET /api/health`: backend status and explicit model connection states.
- `POST /api/uploads/validate`: validates JPG, PNG or WEBP files up to 10 MB. It does not store the photo and does not run prediction or enhancement.

## Model integration plan

1. Skin classifier: train and evaluate separately in Colab; serve a versioned checkpoint through a dedicated inference service.
2. GFPGAN: reuse the existing pretrained GFPGAN v1.3 integration; do not replace the current Streamlit workflow until the API integration is tested.
3. DCGAN: train as a separate research/augmentation experiment. Never use generated images as the held-out test set or present synthetic outputs as real patient images.

## Safety and privacy

This foundation does not diagnose skin conditions, generate medical causes, save photographs, or implement accounts yet. Add authentication, consent, private storage, retention/deletion controls and rate limits before enabling personal history or production uploads. Never commit API keys, passwords, private photographs or model weights.
