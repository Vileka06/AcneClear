import json
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_REPO_ID = os.environ.get("HF_ADAPTER_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not ADAPTER_REPO_ID:
    raise RuntimeError("Set HF_ADAPTER_REPO in Hugging Face Space Settings > Variables.")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in Hugging Face Space Settings > Secrets. It needs read access to your private adapter repository.")

# Verified facts from the AcneClear project and its completed training notebook.
PROJECT_KNOWLEDGE = """
ACNECLEAR AI PROJECT KNOWLEDGE (use these facts when the user asks about this project):
- Project: AcneClear AI, a Streamlit skin-wellness and photo-retouching student application.
- Website modules: Acne Remover, Beauty Tips, Diet Planner, and one unified AI Chatbot.
- Acne Remover: user uploads JPG/PNG/WEBP portrait; the app calls pretrained TencentARC GFPGAN v1.3 (architecture setting 'clean', channel_multiplier=2, upscale=2, restoration weight=0.6), then applies an OpenCV bilateral smoothing pass and a YCrCb-based blending mask. It compares original and retouched photos and permits PNG download. GFPGAN is a general face-restoration model, NOT an acne detector or dedicated acne-removal model. The app does not medically diagnose acne or guarantee removal of acne marks. GFPGAN is pretrained and was integrated; it was not trained from scratch for this project.
- Beauty Tips: collects skin type, concerns, sensitivity and an optional question; the LLM generates simple educational skincare suggestions. It is not a diagnosis or prescription.
- Diet Planner: collects food preference, meal style, budget, number of meals, foods to avoid/allergies and an optional request; the LLM generates general recipe/meal ideas. These are not medical diets or guaranteed acne treatments.
- Unified chatbot: the same fine-tuned Qwen3 model is intended to answer questions about all AcneClear modules and ordinary general topics. It maintains recent conversation history. It does not have live web browsing.
- LLM: base model Qwen/Qwen3-0.6B, a pretrained decoder-only causal language model with 28 Transformer blocks, hidden size 1024, 16 attention heads, 8 key/value heads, intermediate size 3072, vocabulary size 151936. It has approximately 601,096,192 total parameters.
- Fine-tuning: supervised fine-tuning using PEFT LoRA. LoRA rank 8, alpha 16, dropout 0.05. Target projections: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. Trainable adapter parameters: 5,046,272 (0.8395% of the base model parameters). Base model weights were frozen; this is fine-tuning, not pretraining an LLM from scratch.
- Completed training notebook results: 3 epochs; total runtime 62.47 minutes; 2,757 prepared examples (190 acne, 1,369 beauty ingredient/skin-care examples, 1,198 Indian recipe examples); split 2,205 train / 276 validation / 276 test. Held-out test loss 0.7037; test perplexity 2.0212; sampled response-token accuracy 0.7828, measured on 64 test examples (13,860 correct next-token predictions out of 17,705 response tokens); final validation loss 0.7175. These are language-model training/evaluation diagnostics, not clinical accuracy or factual answer accuracy.
- Dataset sources in the notebook: kingabzpro/dermatology-qa-firecrawl-dataset and Mreeb/Dermatology-Question-Answer-Dataset-For-Fine-Tuning for acne; cureskin/indian-skincare-ingredient-flags for beauty/ingredient examples; kanishk307/IndianFoodDatasetGeneration for recipe examples. The notebook prepared the final dataset by filtering and formatting these sources.
- Tech stack: Python, Streamlit, PyTorch, Transformers, PEFT/LoRA, Hugging Face, GFPGAN, OpenCV, NumPy, Pillow, Gradio and gradio_client. Streamlit is the website; Hugging Face Space hosts the LLM inference service.
- Deployment: GitHub repository Vileka06/AcneClear, Streamlit Community Cloud website https://acneclear.streamlit.app/.
- Architecture distinction: GFPGAN handles image restoration. Qwen3 + LoRA handles text responses. They are separate models for separate tasks.
When answering project questions, use these known facts, explain in simple language, and distinguish verified project implementation from possible future improvements. Do not invent model accuracy, extra features, or training results.
"""

SYSTEM_PROMPT = (
    "You are AcneClear AI, a clear, friendly, general-purpose assistant and project guide. "
    "Answer the user's actual question directly in simple language. You can answer general questions "
    "about education, programming, science, maths, writing and everyday topics. When asked about the "
    "AcneClear AI project, use the verified project notes below as the main source of truth. "
    "Explain technical ideas with a short example when helpful. Do not invent facts or claim features "
    "that are not listed. The project notes are context, not a requirement to mention the project in "
    "unrelated answers. For current/live facts, say that you cannot browse the web. "
    "For acne and skincare, provide general education only: do not diagnose, prescribe, recommend "
    "changing medication, or promise a cure. Refer deep, painful, persistent or scarring acne to a "
    "dermatologist. For allergies, pregnancy, medical conditions or medication-related diets, recommend "
    "a qualified clinician or registered dietitian. Do not make unsupported nutrient claims."
)

torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {BASE_MODEL_ID} + {ADAPTER_REPO_ID} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_REPO_ID,
    token=HF_TOKEN,
)
model.to(DEVICE)
model.eval()
print("Fine-tuned AcneClear model is ready.")


def normalize_history(history):
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except json.JSONDecodeError:
            history = []
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content[:1800]})
    return cleaned


def answer(message, history):
    user_text = str(message or "").strip()
    if not user_text:
        return "Please type a question."

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + PROJECT_KNOWLEDGE}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": user_text[:2500]})

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=320,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer_ids = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return response or "I couldn't form a useful answer. Please rephrase your question."


# A two-input Interface gives the Streamlit gradio_client a stable API contract:
# predict(message, history, api_name="/chat").
demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Message"),
        gr.JSON(label="Recent conversation history"),
    ],
    outputs=gr.Textbox(label="AcneClear AI answer"),
    title="AcneClear AI Assistant",
    description="Backend for the AcneClear Streamlit chatbot. It answers general and project-specific questions.",
    examples=[
        ["Explain the four modules in my AcneClear AI project.", []],
        ["What is the difference between GFPGAN and Qwen3?", []],
        ["How many Transformer blocks does Qwen3-0.6B have?", []],
        ["Suggest a gentle routine for oily skin.", []],
        ["Give me affordable vegetarian South Indian meal ideas.", []],
    ],
    api_name="chat",
)
demo.queue(default_concurrency_limit=1).launch()
