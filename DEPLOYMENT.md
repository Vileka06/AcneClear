# AcneClear React + FastAPI deployment

The current Streamlit deployment at https://acneclear.streamlit.app/ is still served by the legacy root-level `app.py`. Do not delete or overwrite it during migration. The new React website and FastAPI backend are deployed as separate services.

## 1. Deploy the API on Render

1. Open https://dashboard.render.com/ and choose New > Blueprint.
2. Connect this GitHub repository: `Vileka06/AcneClear`.
3. Select branch `feature/acneclear-foundation`.
4. Render detects the root `render.yaml`. Review the service and create it.
5. After deployment, copy the service URL, for example `https://acneclear-api.onrender.com`.
6. Check `https://YOUR-API-URL/api/health`.

The service asks for `ACNECLEAR_WEB_ORIGIN`. It can be left blank initially because the backend also allows Vercel preview origins. After the Vercel website is deployed, set this variable to the exact production website origin in Render.

## 2. Deploy the website on Vercel

1. Open https://vercel.com/new and import `Vileka06/AcneClear`.
2. Select branch `feature/acneclear-foundation`.
3. Set Root Directory to `frontend/website`.
4. Framework preset: Vite.
5. Build command: `npm run build`.
6. Output directory: `dist`.
7. Add environment variable:
   - Name: `VITE_API_URL`
   - Value: the Render API URL from step 1, without a trailing slash.
8. Deploy.

Vercel automatically creates preview deployments for branch pushes. Use the production deployment URL to test the website and upload validation.

## 3. Finish the migration

Only after both services have been tested:
- Set `ACNECLEAR_WEB_ORIGIN` in Render to the exact Vercel production origin, e.g. `https://acneclear-ai.vercel.app`.
- Redeploy the Render API.
- Decide whether to use a custom domain for the React website.
- Keep the Streamlit app and original `app.py` as a rollback copy until the new website is fully accepted.

## Important hosting note

The address `acneclear.streamlit.app` is a Streamlit Community Cloud deployment. It cannot directly host a React/Vite frontend with a separate FastAPI backend. The new React website receives a Vercel URL; the FastAPI service receives a Render URL. A custom domain can later point to Vercel. Do not promise that the old Streamlit subdomain will become the Vercel site.
