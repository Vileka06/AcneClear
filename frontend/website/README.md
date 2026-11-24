# AcneClear AI website

The React + Vite website provides one workflow: upload a portrait, request photo enhancement from the FastAPI service, compare the original and enhanced images, and download the PNG.

## Local development

Start the FastAPI backend first. Follow the setup in the repository root README, then:

~~~bash
npm install
cp .env.example .env
npm run dev
~~~

Set VITE_API_URL in the local .env file to the backend URL. For local development the default is http://127.0.0.1:8000.

## Production deployment

Deploy this directory as a Vite project on Vercel:
- Root directory: frontend/website
- Build command: npm run build
- Output directory: dist
- Environment variable: VITE_API_URL set to the deployed FastAPI service URL.

See the repository DEPLOYMENT.md for the full setup steps.
