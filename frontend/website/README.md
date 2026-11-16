# AcneClear AI website

React + Vite frontend for AcneClear AI.

## Local development

```bash
npm install
cp .env.example .env
npm run dev
```

Set `VITE_API_URL` to the FastAPI backend URL. For local development it defaults to `http://localhost:8000`.

## Production deployment

Deploy this directory as a Vite project on Vercel:
- Root directory: `frontend/website`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL=https://YOUR-RENDER-API-URL`

See the repository's `DEPLOYMENT.md` for the full migration steps.
