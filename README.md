# AcneClear 

<p align="center">
  <strong>AI-assisted portrait retouching and skin-wellness guidance</strong><br>
  A Streamlit project with a photo-retouching demo, skincare tips, and balanced meal ideas.
</p>

<p align="center">
  <a href="https://acneclear.streamlit.app/">Open the live demo</a> 
</p>

---

## Project overview

**AcneClear AI** is a student project that brings together a simple portrait-retouching workflow and educational skin-wellness content in one web application. The interface uses a cream, sage-grey, and burgundy colour palette.

## Features

### 1. Acne Remover
- Upload a portrait image in JPG, PNG, or WEBP format.
- Generate a restored and lightly smoothed preview.
- Compare the original and processed images.
- Download the processed result as a PNG.

### 2. Beauty Tips
- General morning and evening skincare routine suggestions.
- Gentle skincare habits and reminders.

### 3. Diet Planner
- General meal ideas for vegetarian, egg-based, and meat/fish preferences.
- Balanced eating suggestions and notes about individual food triggers.

## How the photo editor works

1. The user uploads a portrait.
2. The app passes it to the pretrained **GFPGAN v1.3** face-restoration model.
3. A separate, lightweight OpenCV bilateral-smoothing step is applied.
4. The user compares and downloads the result.

> **Model limitation:** GFPGAN is a pretrained face-restoration model, not an acne-specific detector or remover. The smoothing step is basic image processing and cannot guarantee removal of individual acne marks. The current app does not train a GAN from scratch or perform medical analysis.

## Technology stack

- **Python**
- **Streamlit** — web interface
- **GFPGAN v1.3** — pretrained facial restoration
- **OpenCV and NumPy** — image processing
- **Pillow** — image loading and PNG export
- **Streamlit Community Cloud** — deployment

## Run locally

Recommended Python version: **3.11**.

1. Clone this repository:

   ```bash
   git clone https://github.com/Vileka06/AcneClear.git
   cd AcneClear
   ```

2. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

On first use, the app downloads the official GFPGAN v1.3 checkpoint into the local `models/` directory. The model checkpoint is excluded from Git by `.gitignore`.

## Deploy

The app is deployed on Streamlit Community Cloud:

**Live app:** https://acneclear.streamlit.app/

For deployment changes, connect the `Vileka06/AcneClear` repository, use the `main` branch, and set the main file to `app.py`. The Python version and dependency list are configured for the current deployment; if changing them, test the build logs before considering the deployment complete.

## Repository files

```text
AcneClear/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

## Privacy and responsible use

- Upload only photos you have permission to process.
- Avoid uploading sensitive or private images to a public demo.
- The app is an educational photo-retouching and wellness demo, not a medical device or a substitute for a dermatologist.
- Diet and skincare information is general education and is not intended to diagnose or treat acne.
- Never commit passwords, API keys, environment files, uploaded photos, or model-weight files to the repository.

## Acknowledgements

- [TencentARC GFPGAN](https://github.com/TencentARC/GFPGAN) for pretrained face-restoration technology.
- [Streamlit](https://streamlit.io/) for the web app framework.

## 4. Fine-tuned LLM assistant

The Beauty Tips and Diet Planner sections now use the trained Qwen3-0.6B LoRA adapter through a separate Hugging Face Gradio Space. A general-purpose **AI Chatbot** section is also available.

- **Base LLM:** Qwen/Qwen3-0.6B
- **Fine-tuning:** Supervised fine-tuning with LoRA
- **Website features:** general chatbot, personalized educational skincare guidance, and meal ideas
- **Inference:** Hugging Face Space; Streamlit sends prompts through the Gradio API
- **GFPGAN:** the existing photo-restoration implementation remains separate and unchanged

See [CHATBOT_DEPLOYMENT.md](CHATBOT_DEPLOYMENT.md) for adapter upload, Space setup, and Streamlit Secrets configuration.

The chatbot does not have live web access. Skincare responses are educational, not diagnosis or treatment. Meal suggestions are not medical diets. The fine-tuning test metrics are language-model diagnostics, not medical correctness scores.
