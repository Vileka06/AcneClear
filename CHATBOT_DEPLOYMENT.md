# AcneClear AI — connect your completed Qwen3 LoRA model

Your uploaded Colab notebook confirms that training completed for 3 epochs. The adapter is saved in Google Drive:

`MyDrive/AcneClear_AI/Qwen3_LoRA_Training/qwen3_0_6b_lora_adapter`

The notebook's Hugging Face upload cell was skipped (PUSH_TO_HUB=False), so the adapter is not yet available to the website. Do the steps below once.

## 1. Upload the adapter from Google Drive to Hugging Face Hub

Open the completed Colab notebook. Add a new cell and run:

```python
from google.colab import drive
drive.mount("/content/drive")

from huggingface_hub import notebook_login, HfApi
from pathlib import Path

notebook_login()  # paste a Hugging Face write token when prompted

adapter_dir = Path(
    "/content/drive/MyDrive/AcneClear_AI/Qwen3_LoRA_Training/qwen3_0_6b_lora_adapter"
)
assert (adapter_dir / "adapter_config.json").exists(), "Adapter folder is missing adapter_config.json"
assert any(adapter_dir.glob("adapter_model.*")), "Adapter weights are missing"

repo_id = "YOUR_HF_USERNAME/acneclear-qwen3-lora"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(adapter_dir),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload trained AcneClear Qwen3 LoRA adapter",
)
print("Uploaded adapter:", repo_id)
```

Replace `YOUR_HF_USERNAME` with your actual Hugging Face username. Use a Hugging Face write token only inside Colab's login prompt; never paste it into GitHub or app.py.

## 2. Create a Hugging Face Space for model inference

1. On Hugging Face, create a new **Space**.
2. Choose **Gradio** as the SDK.
3. Choose **Public** for the Space so the Streamlit website can call it. The model adapter itself can remain private.
4. Copy the files from this repository's `chatbot_space/` folder into the root of the new Space:
   - `app.py`
   - `requirements.txt`
   - `README.md`
5. In Space **Settings → Variables and secrets**, add:
   - Variable `HF_ADAPTER_REPO` = `YOUR_HF_USERNAME/acneclear-qwen3-lora`
   - Secret `HF_TOKEN` = Hugging Face read token with access to the private adapter.
6. Wait for the Space to build and become **Running**. First startup downloads the base model and adapter. CPU inference may be slower; use a supported GPU Space if available.

## 3. Connect Streamlit Community Cloud

In the Streamlit app deployment settings → **Secrets**, add:

```toml
ACNECLEAR_CHAT_SPACE = "YOUR_HF_USERNAME/YOUR_SPACE_NAME"
```

If your Gradio Space is private, add a read token too:

```toml
ACNECLEAR_CHAT_TOKEN = "hf_your_read_token"
```

For a public Space, omit `ACNECLEAR_CHAT_TOKEN`.

## 4. Test the website

- AI Chatbot: ask a general question, e.g. "Explain Python functions with an example."
- Beauty Tips: select oily/sensitive skin and request a morning/evening routine.
- Diet Planner: select vegetarian, South Indian, affordable and request meal ideas.
- Acne Remover: upload a test portrait and confirm GFPGAN still works.

## What changed in app.py

- Added an AI Chatbot page.
- Beauty Tips now requests generated skincare guidance from the fine-tuned LLM.
- Diet Planner now requests generated meal ideas from the fine-tuned LLM.
- The existing GFPGAN photo-restoration functions and Acne Remover branch were left unchanged.
- The model runs in the separate Hugging Face Space; Streamlit only calls its Gradio API.

## Model and safety notes

- Base model: Qwen/Qwen3-0.6B.
- Fine-tuning: supervised fine-tuning with LoRA.
- The adapter is the trained project model, not a full copy of Qwen3.
- General questions use the base LLM's existing general capabilities plus your fine-tuning.
- The model has no live web access.
- Skincare answers are educational, not diagnosis or treatment.
- Meal ideas are not medical nutrition prescriptions. For allergies, pregnancy, chronic illness, or medicine-related diets, advise a qualified clinician or dietitian.
- Do not claim that language-model loss/perplexity measures medical correctness.
