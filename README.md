# AcneClear AI

**Live Demo:** [Open AcneClear AI](https://acneclear.streamlit.app/)

A simple college mini-project with two features:

1. **Acne Remover** — upload a portrait, create a softly retouched preview using GFPGAN v1.3 and gentle image smoothing, compare before and after, and download the result.
2. **AI Skincare Chatbot** — ask questions about acne, skincare routines, ingredients, sunscreen and everyday skin wellness. The chatbot uses the Google Gemini API and keeps conversation history for the current app session.

## Technology

- Python and Streamlit
- GFPGAN v1.3, PyTorch, OpenCV and Pillow
- Google Gemini API for the chatbot

## Run locally

```bash
git clone https://github.com/Vileka06/AcneClear.git
cd AcneClear
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key before starting the app:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
streamlit run app.py
```

On macOS/Linux, run these commands in the project terminal. For Streamlit Community Cloud, add the following in **App settings → Secrets**:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Never commit a real API key to GitHub. If the key is missing, the photo enhancement feature remains available and the chatbot displays a setup message.

## Live demo

https://acneclear.streamlit.app/

Streamlit Community Cloud should deploy from the `main` branch with `app.py` as the main file. Reboot the app after updating secrets.

## Model note

The photo feature uses GFPGAN v1.3 for general face restoration with gentle texture smoothing. It is not a dedicated acne-removal, skin-diagnosis or medical-treatment model. The chatbot provides general educational information, not medical advice.
