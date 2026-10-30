import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER_REPO_ID = os.environ.get("HF_ADAPTER_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not ADAPTER_REPO_ID:
    raise RuntimeError("Set HF_ADAPTER_REPO in Hugging Face Space Variables.")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in Hugging Face Space Secrets with read access to your private adapter repository.")

SYSTEM_PROMPT = (
    "You are AcneClear AI, a friendly general-purpose assistant. Answer general questions clearly and helpfully. "
    "You can also explain acne, skincare ingredients, and ordinary Indian recipes. For acne and skin concerns, "
    "give educational information only; do not diagnose, prescribe, recommend changing medicines, or promise a cure. "
    "For deep, painful, persistent, rapidly worsening, or scarring acne, recommend a qualified dermatologist. "
    "For food, allergies, pregnancy, medical conditions, or medicine-related dietary questions, do not create a medical diet; "
    "suggest consulting a registered dietitian or clinician. Recipe examples are not nutrition prescriptions. "
    "Do not invent nutrient values that are not present in the source. If uncertain, say so."
)

torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {BASE_MODEL_ID} with adapter {ADAPTER_REPO_ID} on {DEVICE}...")
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
print("AcneClear fine-tuned model loaded.")


def answer(message, history):
    user_text = str(message).strip()
    if not user_text:
        return "Please type a question."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Retain the most recent four exchanges to keep context and memory usage bounded.
    for item in (history or [])[-8:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content[:1800]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_part, assistant_part = item
            if isinstance(user_part, str) and user_part.strip():
                messages.append({"role": "user", "content": user_part[:1800]})
            if isinstance(assistant_part, str) and assistant_part.strip():
                messages.append({"role": "assistant", "content": assistant_part[:1800]})

    messages.append({"role": "user", "content": user_text[:2500]})
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=260,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response or "I couldn't form a useful answer. Please rephrase your question."


demo = gr.ChatInterface(
    fn=answer,
    type="messages",
    title="AcneClear AI Assistant",
    description="Ask general questions or get educational help with acne, skincare, and Indian recipe ideas.",
    examples=[
        "Explain Python functions with an example.",
        "I have oily and sensitive skin. Suggest a gentle routine.",
        "Suggest affordable vegetarian Indian meal ideas.",
        "What is the difference between RAM and storage?",
    ],
    api_name="chat",
)
demo.launch()
