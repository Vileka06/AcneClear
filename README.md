# AcneClear AI

A simple college mini-project for photo enhancement using a pretrained GAN-based face-restoration model.

## Main feature

- **Photo Enhancement** — upload a clear portrait, restore facial details with GFPGAN v1.3, compare the original and enhanced image, and download the result.

The app intentionally keeps the earlier working, simple photo-enhancement flow. It does not include the chatbot, skin analysis, diet planner, product recommendations, reminders, or the unfinished DCGAN checkpoint upload screen.

## Technology

- Python
- Streamlit
- PyTorch (installed as a dependency of GFPGAN)
- GFPGAN v1.3
- OpenCV and Pillow

## Run locally

```bash
git clone https://github.com/Vileka06/AcneClear.git
cd AcneClear
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501.

## Live demo deployment

Streamlit Community Cloud settings:

- Repository: `Vileka06/AcneClear`
- Branch: `main`
- Main file: `app.py`

After the GitHub update, open the Streamlit app dashboard and select **Reboot app** if it does not redeploy automatically.

## Model note

On first use, the app downloads the official GFPGAN v1.3 weights into the local `models/` directory. The model is a general face-restoration GAN, not a dedicated acne-removal or medical diagnosis model. Results depend on photo quality and face visibility.
