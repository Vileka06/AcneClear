# AcneClear AI

**Live demo:** https://acneclear.streamlit.app/

AcneClear AI is a simple Streamlit application with only two features:

1. **Acne Remover** – upload a portrait, create a softly retouched preview, compare before and after, and download the result.
2. **AI Chatbot** – ask questions about acne, skincare routines, ingredients, sunscreen and everyday skin wellness.

## Technology

- Python and Streamlit
- GFPGAN v1.3 for general facial restoration
- OpenCV for gentle skin-texture smoothing
- Google Gemini API (`gemini-3.5-flash-lite`) for the skincare chatbot

**Important photo limitation:** GFPGAN is a general face-restoration model, not a dedicated acne detector or guaranteed acne-removal model. The app combines facial restoration with gentle texture smoothing. Results vary, and this is not a medical treatment.

## Run locally

Use Python 3.11.

```bash
git clone https://github.com/Vileka06/AcneClear.git
cd AcneClear

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Enable the chatbot

1. Create a Gemini API key at https://aistudio.google.com/apikey.
2. In Streamlit Community Cloud, open the app's **Settings → Secrets**.
3. Add the following secret:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

4. Save the settings and reboot the app if needed.

For local use, create `.streamlit/secrets.toml` with the same setting. Never commit this file or expose the key in source code. The app also supports the `GEMINI_MODEL` environment variable if you need to select a model available to your API key.

## Streamlit deployment

Keep these settings for the existing live demo:

- Repository: `Vileka06/AcneClear`
- Branch: `feature/acneclear-foundation`
- Main file: `app.py`
- Live URL: https://acneclear.streamlit.app/

The existing Streamlit Community Cloud app should rebuild after the GitHub commit. If it does not, open the app in the Streamlit dashboard and choose **Reboot app**.

## Project files

```text
AcneClear/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

## Safety note

The chatbot provides general educational information only. It does not diagnose skin conditions, prescribe medication or replace a dermatologist. For deep, painful, persistent or scarring acne, seek professional advice.
