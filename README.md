# AcneClear AI

AcneClear AI is a Streamlit-based educational skin-wellness demo with three sections:

- **Acne Remover:** Upload a portrait, run pretrained GFPGAN face restoration with a light image-smoothing pass, compare original and processed images, and download the result.
- **Beauty Tips:** General morning and evening skincare guidance.
- **Diet Planner:** General balanced meal suggestions for different food preferences.

## Important limitations

The photo editor uses **pretrained GFPGAN v1.3**, a face-restoration model, not a model specifically trained to detect or remove acne. The additional smoothing effect is basic image processing and may not remove individual blemishes. Beauty and diet content is educational, not medical advice.

## Run locally

Recommended Python: 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\\Scripts\\activate`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The official GFPGAN checkpoint is downloaded automatically on first use. Do not commit model weights or your virtual environment to GitHub.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app**, select this repository and the `main` branch, and set the main file path to `app.py`.
3. Deploy. Startup may take longer while dependencies install and the model downloads.

If deployment fails, open **Manage app → Logs** and inspect the first dependency/import error.

## Privacy and safety

- Do not upload photos you do not have permission to use.
- This demo is not a medical device and does not diagnose, prevent, or treat acne.
- Do not commit credentials, API keys, model weights, or private images to the repository.
