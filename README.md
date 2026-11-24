# AcneClear AI

A simple college mini project demonstrating two GAN-based image features:

1. **Generate Photo** – create a new image using a trained DCGAN generator checkpoint.
2. **Enhance Photo** – restore facial details in an uploaded portrait using pretrained GFPGAN.

No chatbot, skin analysis, diet planner, product recommendations, reminders, or unrelated modules are included.

## Technology

- Python 3.11
- Streamlit
- PyTorch
- DCGAN generator architecture (64 × 64 output; trained checkpoint supplied by the project owner)
- GFPGAN v1.3 for general face restoration
- OpenCV and Pillow

## Run locally

```bash
git clone https://github.com/Vileka06/AcneClear.git
cd AcneClear
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## Generate Photo: required checkpoint

The trained DCGAN checkpoint is not committed to this repository. In the **Generate Photo** tab, upload your trained `.pth` or `.pt` file.

The current loader expects a standard DCGAN generator with:
- latent vector size: 100
- RGB output: 3 channels
- image size: 64 × 64
- base feature size: 64
- a PyTorch generator state dictionary (or a dictionary containing `generator_state_dict`, `state_dict`, `generator`, or `G`)

If your Colab model uses a different architecture or layer names, adapt `DCGANGenerator` in `app.py` to match your training notebook before using its checkpoint.

## Enhance Photo

The first enhancement request downloads the official GFPGAN v1.3 weights into the local `models/` folder. The weights are ignored by Git and are not uploaded to the repository.

GFPGAN is a general face-restoration GAN. It does not specifically detect or guarantee removal of acne.

## Streamlit Community Cloud deployment

Use these settings in Streamlit Community Cloud:

- Repository: `Vileka06/AcneClear`
- Branch: `main`
- Main file: `app.py`

After pushing changes, reboot the app from the Streamlit dashboard if it does not rebuild automatically.

## Notes

- This project is a demonstration, not a medical application.
- Keep model weights out of Git if they are large. Upload the generator checkpoint in the app when testing.
