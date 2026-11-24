# AcneClear AI

**AcneClear AI is now a focused photo-enhancement project.** Upload a portrait, restore facial details with pretrained GFPGAN v1.3, preview the result and download the enhanced PNG.

The website, Streamlit app and Expo mobile starter expose only this photo-enhancement workflow. Skin analysis, beauty tips, diet planning, chatbot and reminder screens are not part of the current product.

## What it does

- Accepts JPG, PNG and WEBP face photos up to 10 MB.
- Uses pretrained TencentARC GFPGAN v1.3 for general face restoration.
- Applies a restrained OpenCV bilateral-smoothing blend.
- Displays original and enhanced images.
- Returns a downloadable PNG on the website and Streamlit app.
- Processes uploaded photos in memory; the application does not save uploaded photos.

**Important limitation:** GFPGAN is a general face-restoration model, not an acne-specific detector or guaranteed acne-removal model. It may improve facial detail and soften some texture, but individual marks can remain and results vary with lighting, focus and face visibility. AcneClear is not a medical diagnosis or treatment service.

## Project structure

~~~text
AcneClear/
├── app.py                    # Single-feature Streamlit app
├── requirements.txt          # Streamlit + GFPGAN dependencies
├── backend/
│   ├── main.py               # FastAPI photo-enhancement endpoint
│   └── requirements.txt
├── frontend/website/         # React + Vite web client
├── mobile/                   # Expo React Native client
├── render.yaml               # Render API service configuration
└── README.md
~~~

Model weights are downloaded on first use into a local models directory and are excluded from Git. Do not commit model weights, API keys or private photos.

## Run locally

Use Python 3.11 for the backend/model environment.

### 1. Start the FastAPI backend

~~~bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
~~~

The first enhancement request downloads the official GFPGAN v1.3 checkpoint. Check the API at http://127.0.0.1:8000/api/health; interactive API documentation is at http://127.0.0.1:8000/docs.

### 2. Start the React website

In a second terminal:

~~~bash
cd frontend/website
npm install
~~~

Create frontend/website/.env with:

~~~env
VITE_API_URL=http://127.0.0.1:8000
~~~

Then run:

~~~bash
npm run dev
~~~

Open http://127.0.0.1:5173.

### 3. Run the Streamlit version (optional)

From the repository root, create a Python 3.11 environment and install the root dependencies:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

### 4. Run the mobile app (optional)

~~~bash
cd mobile
npm install
~~~

Set EXPO_PUBLIC_API_URL to the reachable FastAPI URL. For a physical phone, use your computer's LAN IP address rather than localhost. Then run npx expo start.

## API

- GET /api/health — API and model-load status.
- POST /api/enhance — accepts a multipart field named file and returns image_base64, content_type and a processing note.

Supported image uploads are limited to 10 MB. Photos are read in memory and are not written to an upload directory.

## Deployment

The Render blueprint in render.yaml deploys the FastAPI service from the backend directory. The React website can be deployed separately to Vercel with root directory frontend/website, build command npm run build, output directory dist, and VITE_API_URL set to the deployed API URL.

The GFPGAN/PyTorch dependencies are large and inference is CPU-intensive. Check Render build logs, available memory and the first model-load request before treating a free-tier deployment as production-ready. The existing Streamlit URL https://acneclear.streamlit.app/ continues to serve the root-level app.py if that deployment remains connected to this branch.

## Acknowledgements

- TencentARC GFPGAN for pretrained face-restoration technology.
- OpenCV for image processing.
- Streamlit, FastAPI, React and Expo for application frameworks.
