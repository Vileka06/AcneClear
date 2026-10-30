# Connect your trained LLM to AcneClear AI

The Streamlit website code is already committed. The chatbot will not answer until the fine-tuned adapter is uploaded to Hugging Face and the Space is deployed.

## A. Upload your completed adapter from Google Drive

Open your completed Colab notebook. It already mounted Google Drive. Add a new cell and run:

\`\`\`python
!pip -q install -U huggingface_hub

from huggingface_hub import notebook_login, HfApi
from pathlib import Path

notebook_login()  # paste a Hugging Face WRITE token in the login prompt

adapter_dir = Path(
    "/content/drive/MyDrive/AcneClear_AI/Qwen3_LoRA_Training/qwen3_0_6b_lora_adapter"
)
assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"
assert any(adapter_dir.glob("adapter_model.*")), "Missing adapter_model weights"

repo_id = "YOUR_HF_USERNAME/acneclear-qwen3-lora"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(adapter_dir),
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload completed AcneClear Qwen3 LoRA adapter",
)
print("Adapter uploaded:", repo_id)
\`\`\`

Replace \`YOUR_HF_USERNAME\` with your Hugging Face username. The token needs **write** permission for this upload. Do not paste the token into GitHub, app.py, or chat.

## B. Create the Hugging Face Space

1. Go to Hugging Face → Spaces → Create new Space.
2. Choose Gradio SDK. A public Space is easiest for Streamlit to call; keep the model adapter repository private if you prefer.
3. Name it, for example, \`acneclear-ai-assistant\`.
4. Open the GitHub repository folder \`chatbot_space/\` and upload its three files into the **root** of the new Space:
   - \`app.py\`
   - \`requirements.txt\`
   - \`README.md\`
5. In Space **Settings → Variables and secrets**, add:
   - Variable \`HF_ADAPTER_REPO\` = \`YOUR_HF_USERNAME/acneclear-qwen3-lora\`
   - Secret \`HF_TOKEN\` = a Hugging Face **read** token with access to that private model repository.
6. Wait until the Space says **Running**. On first startup, it downloads Qwen3-0.6B and your adapter. Free CPU inference can be slow; if memory is insufficient, use a supported GPU Space or a smaller quantized serving setup.

The Space exposes \`/chat\` with two API inputs: message and conversation history.

## C. Connect Streamlit Community Cloud

Open the Streamlit Community Cloud app settings → **Secrets** and add:

\`\`\`toml
ACNECLEAR_CHAT_SPACE = "YOUR_HF_USERNAME/acneclear-ai-assistant"
\`\`\`

For a public Space, no additional Space token is needed. For a private Space, add:

\`\`\`toml
ACNECLEAR_CHAT_TOKEN = "hf_your_read_token"
\`\`\`

Save the secrets. Streamlit will restart. The Diet Planner, Beauty Tips and AI Chatbot all use this same endpoint/model.

## D. Test the unified chatbot

Open https://acneclear.streamlit.app/ and choose **AI Chatbot**. Try:

1. "Explain all four modules of my AcneClear AI project."
2. "What is the difference between GFPGAN and Qwen3?"
3. "How many Transformer blocks does my LLM have?"
4. "Explain my training dataset and results."
5. "Explain Python functions with an example."

Also test Beauty Tips and Diet Planner. Both should generate their answers using the same LLM.

## Why your screenshots did not show answers

The Diet Planner screenshot shows this exact error: \`ACNECLEAR_CHAT_SPACE\` is missing from Streamlit Secrets. The chatbot screenshot shows an empty conversation; no answer can be generated until the Space is deployed and connected. The adapter was saved in Google Drive but the notebook's Hub upload cell was skipped.

## Project facts embedded in the chatbot

The Space system context includes the verified project structure and notebook results: GFPGAN v1.3 image restoration; OpenCV smoothing; Qwen3-0.6B with 28 Transformer blocks; LoRA rank 8, alpha 16, dropout 0.05; 5,046,272 trainable adapter parameters; 2,757 prepared examples (190 acne, 1,369 skincare/ingredient, 1,198 Indian recipes); 3 epochs; 62.47 minutes; test loss 0.7037; perplexity 2.0212; sampled response-token accuracy 78.28% on 64 examples; final validation loss 0.7175. Accuracy/loss/perplexity are language-model diagnostics, not medical or answer-quality guarantees.

The unified chatbot can answer general questions using Qwen's base-model capabilities, but it has no live web access. For current facts it should say that it cannot browse. Skincare advice is educational, not diagnosis or treatment; diet ideas are not medical nutrition prescriptions.
