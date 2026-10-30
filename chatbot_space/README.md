---
title: AcneClear AI Assistant
emoji: 🧴
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# AcneClear AI Assistant

Gradio API backend for the AcneClear Streamlit website.

## Space configuration

Add the following in **Settings → Variables and secrets**:

- Variable: `HF_ADAPTER_REPO` = `YOUR_HF_USERNAME/acneclear-qwen3-lora`
- Secret: `HF_TOKEN` = a Hugging Face read token that can access the private adapter repository.

The Space loads Qwen3-0.6B and attaches the trained LoRA adapter. Keep the adapter repository private if desired. The Space itself can be public so the Streamlit website can call its API.

The ChatInterface exposes the `/chat` endpoint.
