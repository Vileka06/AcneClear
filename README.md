# AcneClear 

<p align="center">
  <strong>AI-assisted portrait retouching and skin-wellness guidance</strong><br>
  A Streamlit project with pretrained face restoration, AI-generated skincare guidance, meal planning, and a general-purpose chatbot.
</p>

<p align="center">
  <a href="https://acneclear.streamlit.app/">Open the live demo</a> 
</p>

---

## Project overview

**AcneClear AI** is a student project that brings together a simple portrait-retouching workflow and educational skin-wellness content in one web application. The interface uses a clean white, blue-grey, and navy colour palette.

## Features

### 1. Acne Remover
- Upload a portrait image in JPG, PNG, or WEBP format.
- Generate a restored and lightly smoothed preview.
- Compare the original and processed images.
- Download the processed result as a PNG.

### 2. Beauty Tips
- Collects skin type, concerns, and sensitivity.
- Uses the live AI assistant to generate general educational morning and evening skincare suggestions.
- Includes safety guidance and does not diagnose or prescribe.

### 3. Diet Planner
- Collects food preference, Indian meal style, budget, meals, and foods to avoid.
- Uses the live AI assistant to generate general meal ideas.
- Does not create medical diets or promise that food cures acne.

### 4. AI Chatbot
- General-purpose chat for education, programming, science, writing, and everyday questions.
- Project-aware answers about AcneClear architecture, GFPGAN, Qwen3, LoRA, datasets, training, metrics, and deployment.
- Maintains recent conversation context. No live web browsing is provided.

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

## 4. AI assistant and fine-tuning experiment

- **Live website chatbot:** Gemini 3.5 Flash-Lite using the Google AI Studio Gemini API free tier.
- **Beauty Tips and Diet Planner:** use the same server-side AI assistant to produce educational routines and general meal ideas.
- **Separate model-training experiment:** Qwen3-0.6B fine-tuned with LoRA in Google Colab. The saved adapter is a project training artifact and is not the online inference model in the current free website deployment.
- **Acne Remover:** continues to use pretrained GFPGAN v1.3 and the existing OpenCV image-processing steps.

See [CHATBOT_DEPLOYMENT.md](CHATBOT_DEPLOYMENT.md) for the free API-key setup. Free-tier access and quotas depend on the Google AI Studio project/account. Free-tier requests may be used to improve Google's products; do not send sensitive personal information.
