# AcneClear photo-enhancement deployment

The repository contains a React website, a FastAPI enhancement API and a Streamlit version of the same single-feature application.

## 1. Deploy the API on Render

1. Open https://dashboard.render.com/ and create a new Blueprint.
2. Connect Vileka06/AcneClear.
3. Select branch feature/acneclear-foundation.
4. Render reads render.yaml; the API root directory is backend.
5. After deployment, test https://YOUR-API-URL/api/health.

Set ACNECLEAR_WEB_ORIGIN to the exact production website origin, such as https://your-site.vercel.app.

**Hosting note:** GFPGAN requires PyTorch and image-processing dependencies, and CPU inference can use significant memory and time. Check build logs, service memory and a real enhancement request. The free service may not be sufficient for reliable production inference.

## 2. Deploy the website on Vercel

1. Import Vileka06/AcneClear.
2. Choose branch feature/acneclear-foundation.
3. Set Root Directory to frontend/website.
4. Choose the Vite framework preset.
5. Set build command to npm run build and output directory to dist.
6. Add VITE_API_URL with the Render API URL (no trailing slash).
7. Deploy, then test a real photo enhancement.

## 3. Streamlit deployment

The existing https://acneclear.streamlit.app/ address serves the root-level app.py when that Streamlit app is connected to this repository and branch. The root requirements.txt contains the Streamlit and GFPGAN dependencies.

## 4. Privacy and limitations

Uploaded photos are processed in memory and are not saved by the application. Model weights are downloaded separately and are excluded from Git. GFPGAN is a general face-restoration model, not an acne-specific detector or guaranteed acne-removal model. Test the deployment with non-sensitive sample photos before sharing it publicly.
