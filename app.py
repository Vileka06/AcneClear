import base64
import io
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# AcneClear AI - photo retouching + skincare education studio
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "models"
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHT_PATH = WEIGHTS_DIR / "GFPGANv1.3.pth"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

_GFPGAN_RESTORER = None
_ACNECLEAR_CHAT_CLIENT = None

# Streamlit cache decorators are intentionally avoided here because source inspection
# fails on this deployed script; the model/client are created only when first requested.

st.set_page_config(
    page_title="AcneClear AI | Skin Wellness Studio",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state defaults
for key, value in {
    "acneclear_original": None,
    "acneclear_result": None,
    "acneclear_chat_history": [],
    "acneclear_beauty_response": None,
    "acneclear_diet_response": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Palette: cream beige, muted sage grey, deep burgundy
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --cream:#F3EBDD; --sage:#A3ACA1; --burgundy:#651F32;
  --ink:#342B2B; --muted:#756B68; --line:#D8D0C6; --white:#FFFFFF;
}
.stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1320px; padding-top:1.3rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:rgba(243,235,221,.96); }
#MainMenu, footer {visibility:hidden;}
h1,h2,h3 {font-family:'Manrope',sans-serif !important; color:var(--burgundy) !important;}
p, label, .stMarkdown { color:var(--ink); }
.brand-row {display:flex;align-items:center;gap:12px;margin:0 0 24px 0;}
.brand-mark {width:44px;height:44px;display:flex;align-items:center;justify-content:center;border-radius:12px;background:var(--burgundy);color:var(--cream);font-size:23px;font-weight:800;}
.brand-name {font-family:'Manrope',sans-serif;font-size:22px;font-weight:800;letter-spacing:-.5px;color:var(--burgundy);line-height:1.1;}
.brand-sub {color:var(--muted);font-size:12px;margin-top:4px;}
.eyebrow {color:var(--burgundy);text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;margin-bottom:9px;}
.hero-title {font-family:'Manrope',sans-serif;color:var(--burgundy);font-weight:800;letter-spacing:-1.3px;font-size:clamp(30px,4vw,46px);line-height:1.08;margin:0;}
.hero-copy {color:var(--muted);font-size:15px;line-height:1.7;max-width:760px;margin-top:12px;}
.panel {background:var(--white);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 8px 26px rgba(65,42,35,.04);}
.panel-title {color:var(--burgundy);font-family:'Manrope',sans-serif;font-weight:800;font-size:17px;margin-bottom:5px;}
.panel-sub {color:var(--muted);font-size:13px;line-height:1.55;margin-bottom:12px;}
.pill {display:inline-block;border:1px solid var(--sage);background:#E7EAE5;color:var(--burgundy);border-radius:6px;padding:5px 9px;font-size:11px;font-weight:700;}
div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
 background:var(--burgundy);color:#fff;border:1px solid var(--burgundy);border-radius:8px;min-height:42px;font-weight:700;padding:.55rem 1rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {background:#7B2A42 !important;border-color:#7B2A42 !important;color:#fff !important;}
div.stButton > button p, div.stDownloadButton > button p, div[data-testid="stFormSubmitButton"] > button p {color:#fff !important;}
div[data-baseweb="input"] input, div[data-baseweb="select"] input, div[data-baseweb="textarea"] textarea {background:#fff !important;color:#342B2B !important;}
div[data-baseweb="select"] > div {background:#fff !important;color:#342B2B !important;}
div[data-testid="stImage"] img {border-radius:14px;}
div[data-testid="stFileUploader"] {background:#fff;border:1px dashed var(--sage);border-radius:12px;padding:10px;}
[data-testid="stSidebar"] {background:#F8F3EA;border-right:1px solid var(--line);}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:var(--burgundy);font-family:'Manrope',sans-serif;}
[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;}
div[data-baseweb="tab-list"] {gap:8px;}
button[data-baseweb="tab"] {background:#E7EAE5;border-radius:8px 8px 0 0;color:var(--burgundy);padding:10px 16px;}
button[data-baseweb="tab"][aria-selected="true"] {background:var(--burgundy);color:#fff;}
.stSlider [data-baseweb="slider"] div {background:var(--burgundy);}
[data-testid="stChatInput"] textarea {background:#fff !important;color:#342B2B !important;}
[data-testid="stChatInput"] textarea::placeholder {color:#756B68 !important;opacity:1 !important;}
[data-testid="stChatInput"] button {color:#651F32 !important;}
.small-note {color:var(--muted);font-size:11px;line-height:1.6;}
hr {border-color:var(--line);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-row">
  <div class="brand-mark">✳</div>
  <div>
    <div class="brand-name">AcneClear AI</div>
    <div class="brand-sub">Skin Wellness Studio · Photo retouching and everyday guidance</div>
  </div>
  <div style="margin-left:auto"><span class="pill">SKIN WELLNESS</span></div>
</div>
<div class="eyebrow">AI-assisted skin wellness</div>
<h1 class="hero-title">Feel good in your skin.<br>Keep your edits natural.</h1>
<p class="hero-copy">Retouch a portrait, explore gentle skincare habits, plan balanced meals, and track your routine in one simple workspace.</p>
""", unsafe_allow_html=True)

def load_restorer():
    global _GFPGAN_RESTORER
    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER

    # Compatibility shim for BasicSR releases that import the removed
    # torchvision.transforms.functional_tensor module.
    import sys
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)

    from gfpgan import GFPGANer
    if not WEIGHT_PATH.exists():
        with st.spinner("Downloading the official GFPGAN v1.3 model (first run only)…"):
            urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
    _GFPGAN_RESTORER = GFPGANer(
        model_path=str(WEIGHT_PATH),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return _GFPGAN_RESTORER

def skin_retouch(image_rgb: np.ndarray, strength: int = 35) -> np.ndarray:
    """A light, conservative smoothing pass; not a dedicated acne-removal model."""
    if strength <= 0:
        return image_rgb
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(
        bgr, d=9, sigmaColor=30 + strength * 1.25, sigmaSpace=30 + strength * 1.25
    )
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array([0, 128, 70], dtype=np.uint8),
        np.array([255, 180, 145], dtype=np.uint8),
    )
    mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32) / 255.0
    mask = (mask * (strength / 100.0) * 0.72)[:, :, None]
    out = bgr.astype(np.float32) * (1.0 - mask) + smooth.astype(np.float32) * mask
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

def compare_slider(before: Image.Image, after: Image.Image, height: int = 520):
    def data_uri(img):
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode("ascii")
    before_uri, after_uri = data_uri(before), data_uri(after)
    html = f"""<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#F3EBDD;font-family:Arial,sans-serif}}
.compare{{position:relative;width:100%;height:{height}px;overflow:hidden;border-radius:12px;background:#fff;border:1px solid #D8D0C6;--pos:50%}}
.compare img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff}}
.after{{clip-path:inset(0 0 0 var(--pos))}}
.line{{position:absolute;top:0;bottom:0;left:var(--pos);width:2px;background:#fff;box-shadow:0 0 0 1px #651F3240;pointer-events:none}}
.knob{{position:absolute;left:var(--pos);top:50%;transform:translate(-50%,-50%);width:38px;height:38px;border-radius:50%;background:#651F32;border:2px solid #fff;box-shadow:0 2px 10px #0002;display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px;pointer-events:none}}
input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
.label{{position:absolute;top:12px;padding:7px 10px;border-radius:5px;background:#651F32;color:#fff;font-size:11px;font-weight:700;letter-spacing:.3px}}
.l{{left:12px}}.r{{right:12px}}
</style></head><body><div class="compare" id="compare"><img src="{before_uri}" alt="Original"><img class="after" src="{after_uri}" alt="Retouched"><div class="line"></div><div class="knob">↔</div><div class="label l">ORIGINAL</div><div class="label r">RETOUCHED</div><input id="range" type="range" min="0" max="100" value="50" aria-label="Compare images"></div>
<script>const c=document.getElementById('compare'),r=document.getElementById('range');r.addEventListener('input',()=>c.style.setProperty('--pos',r.value+'%'));</script>
</body></html>"""
    import streamlit.components.v1 as components
    components.html(html, height=height + 12, scrolling=False)


def _acneclear_setting(name: str) -> str:
    """Read a setting from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def get_acneclear_chat_client():
    """Connect to the dedicated Hugging Face Gradio Space hosting the fine-tuned LLM."""
    global _ACNECLEAR_CHAT_CLIENT
    if _ACNECLEAR_CHAT_CLIENT is not None:
        return _ACNECLEAR_CHAT_CLIENT

    space_id = _acneclear_setting("ACNECLEAR_CHAT_SPACE")
    if not space_id:
        raise RuntimeError(
            "The AI service is not connected yet. Add ACNECLEAR_CHAT_SPACE to "
            "Streamlit Community Cloud → Settings → Secrets after deploying the Hugging Face Space."
        )
    from gradio_client import Client
    token = _acneclear_setting("ACNECLEAR_CHAT_TOKEN")
    if token:
        _ACNECLEAR_CHAT_CLIENT = Client(space_id, token=token, verbose=False)
    else:
        _ACNECLEAR_CHAT_CLIENT = Client(space_id, verbose=False)
    return _ACNECLEAR_CHAT_CLIENT

def ask_acneclear_ai(prompt: str, history=None) -> str:
    """Send a message and the recent chat history to the fine-tuned AcneClear LLM."""
    client = get_acneclear_chat_client()
    result = client.predict(prompt, history or [], api_name="/chat")
    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    if isinstance(result, dict):
        result = result.get("answer", result.get("output", ""))
    return str(result).strip() or "I could not generate a useful answer. Please rephrase your question."


# Sidebar navigation
with st.sidebar:
    st.markdown("## AcneClear AI")
    page = st.radio(
        "Choose a section",
        ["Acne Remover", "Beauty Tips", "Diet Planner", "AI Chatbot"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**About this demo**")
    st.caption("Photo editing is a visual effect, not a medical diagnosis. Skincare and nutrition guidance is educational.")

if page == "Acne Remover":
    upload_col, info_col = st.columns([1.1, 0.9], gap="large")
    with upload_col:
        st.markdown('<div class="panel-title">01 / Upload your portrait</div><div class="panel-sub">Use a clear, well-lit photo with the face visible.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a JPG, PNG or WEBP photo", type=["jpg", "jpeg", "png", "webp"], key="portrait_upload")
        if uploaded is not None:
            try:
                original = Image.open(uploaded).convert("RGB")
                st.session_state["acneclear_original"] = original
                st.image(original, caption="Original photo", use_container_width=True)
            except Exception:
                st.error("This file could not be opened. Please try a JPG or PNG image.")
    with info_col:
        st.markdown('<div class="panel-title">02 / Create your result</div><div class="panel-sub">No settings to adjust. Generate a preview, then compare it with your original.</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">AUTOMATIC RETOUCHING</div><div class="panel-title">Restore details + soften skin</div><div class="panel-sub" style="margin-bottom:0">GFPGAN restores facial details and a light smoothing pass softens some texture. Individual acne marks may remain.</div></div>', unsafe_allow_html=True)
        generate = st.button("Generate retouched photo", type="primary", use_container_width=True, disabled=st.session_state["acneclear_original"] is None)
        reset = st.button("Reset photo and result", use_container_width=True)
        if reset:
            st.session_state["acneclear_original"] = None
            st.session_state["acneclear_result"] = None
            st.rerun()
    if generate and st.session_state["acneclear_original"] is not None:
        with st.spinner("Processing your photo… the first run can take longer."):
            try:
                original = st.session_state["acneclear_original"]
                rgb = np.array(original)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                restorer = load_restorer()
                _, _, restored_bgr = restorer.enhance(
                    bgr, has_aligned=False, only_center_face=False,
                    paste_back=True, weight=0.6
                )
                if restored_bgr is None:
                    raise RuntimeError("The model could not restore this photo. Try a clearer portrait.")
                restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(skin_retouch(restored_rgb, 35))
                st.session_state["acneclear_result"] = result
                st.success("Your preview is ready.")
            except AssertionError:
                st.warning("Face restoration could not align this photo, so a light smoothing-only preview was created.")
                st.session_state["acneclear_result"] = Image.fromarray(skin_retouch(np.array(st.session_state["acneclear_original"]), 35))
            except Exception as exc:
                st.error("Could not process this photo. Check the terminal for model or dependency errors.")
                st.code(f"{type(exc).__name__}: {exc}")
    if st.session_state["acneclear_result"] is not None and st.session_state["acneclear_original"] is not None:
        st.markdown("---")
        st.markdown('<div class="eyebrow">03 / Your result</div><h2>Compare your photo</h2>', unsafe_allow_html=True)
        before = st.session_state["acneclear_original"]
        after = st.session_state["acneclear_result"]
        compare_slider(before, after, 540)
        original_col, result_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("**Original**")
            st.image(before, use_container_width=True)
        with result_col:
            st.markdown("**Retouched result**")
            st.image(after, use_container_width=True)
        out = io.BytesIO()
        after.save(out, format="PNG")
        st.download_button("Download retouched photo (PNG)", out.getvalue(), "acneclear_retouched.png", "image/png", use_container_width=True)

elif page == "Beauty Tips":
    st.markdown(
        '<div class="eyebrow">THE DAILY SKIN EDIT</div>'
        '<h1 class="hero-title">Skincare guidance that starts with you.</h1>'
        '<p class="hero-copy">Tell AcneClear AI a little about your skin and ask a skincare question. The fine-tuned LLM will create an educational routine suggestion.</p>',
        unsafe_allow_html=True,
    )
    st.image(
        "https://images.unsplash.com/photo-1608248543803-ba4f8c70ae0b?auto=format&fit=crop&w=1600&q=85",
        caption="A simple, gentle skincare routine",
        use_container_width=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        skin_type = st.selectbox(
            "Skin type",
            ["Oily", "Dry", "Combination", "Normal", "Sensitive", "Not sure"],
            key="beauty_skin_type",
        )
        concerns = st.multiselect(
            "Main concerns",
            ["Acne-prone skin", "Dryness", "Oiliness", "Uneven tone", "Sensitivity", "Blackheads", "General routine"],
            default=["Acne-prone skin"],
            key="beauty_concerns",
        )
    with right:
        sensitivity = st.selectbox(
            "Skin sensitivity",
            ["Sensitive / easily irritated", "Not especially sensitive", "Not sure"],
            key="beauty_sensitivity",
        )
        extra_question = st.text_area(
            "Ask a beauty or skincare question",
            placeholder="Example: Can I use moisturizer if my skin is oily?",
            height=110,
            key="beauty_question",
        )
    if st.button("Generate my skincare guidance", type="primary", use_container_width=True):
        concern_text = ", ".join(concerns) if concerns else "General skincare"
        prompt = (
            "[BEAUTY TOPIC]\n"
            f"My skin type is {skin_type}. My concerns are {concern_text}. "
            f"My sensitivity level is {sensitivity}. "
            f"Please give me a simple, gentle morning and evening skincare routine. "
            f"Also answer this question if provided: {extra_question or 'No additional question.'} "
            "Use simple steps. Do not diagnose, prescribe medicines, promise acne removal, or recommend many products."
        )
        with st.spinner("AcneClear AI is preparing your skincare guidance..."):
            try:
                st.session_state["acneclear_beauty_response"] = ask_acneclear_ai(prompt)
            except Exception as exc:
                st.error(f"Could not reach the AI service: {type(exc).__name__}: {exc}")
    if st.session_state.get("acneclear_beauty_response"):
        st.markdown("### Your AI-generated skincare guidance")
        st.markdown(st.session_state["acneclear_beauty_response"])
    st.info(
        "This is general skincare education, not a diagnosis or treatment plan. "
        "For painful, persistent, worsening, or scarring acne, consult a dermatologist."
    )

elif page == "Diet Planner":
    st.markdown(
        '<div class="eyebrow">THE EVERYDAY FOOD GUIDE</div>'
        '<h1 class="hero-title">Meal ideas made around your preferences.</h1>'
        '<p class="hero-copy">Choose your food preference and ask the fine-tuned AcneClear LLM for balanced, practical Indian meal ideas.</p>',
        unsafe_allow_html=True,
    )
    st.image(
        "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=1600&q=85",
        caption="A colourful, balanced plate",
        use_container_width=True,
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        preference = st.selectbox(
            "Food preference",
            ["Vegetarian", "Includes eggs", "Includes meat or fish", "Vegan"],
            key="diet_preference",
        )
        meal_style = st.selectbox(
            "Meal style",
            ["South Indian", "North Indian", "Mixed Indian", "Any cuisine"],
            key="diet_meal_style",
        )
    with col2:
        budget = st.selectbox(
            "Budget",
            ["Affordable / student budget", "Moderate", "Flexible"],
            key="diet_budget",
        )
        meals_per_day = st.selectbox(
            "Meals to plan",
            ["3 meals", "3 meals + 1 snack", "3 meals + 2 snacks"],
            key="diet_meals_per_day",
        )
    allergies = st.text_input(
        "Food allergies or foods to avoid (optional)",
        placeholder="Example: peanuts, lactose, mushrooms",
        key="diet_allergies",
    )
    diet_question = st.text_area(
        "What would you like the meal planner to consider?",
        placeholder="Example: I stay in a hostel and need quick meals.",
        height=90,
        key="diet_question",
    )
    if st.button("Generate my meal plan", type="primary", use_container_width=True):
        prompt = (
            "[DIET TOPIC]\n"
            f"Create general Indian meal ideas for this preference: {preference}. "
            f"Meal style: {meal_style}. Budget: {budget}. Meals: {meals_per_day}. "
            f"Allergies/foods to avoid: {allergies or 'None specified'}. "
            f"Additional request: {diet_question or 'No additional request.'} "
            "Give practical breakfast, lunch, snack and dinner suggestions as appropriate. "
            "Do not invent calorie or nutrient values. Do not create a medical diet, crash diet, "
            "or promise that food will cure acne. Respect allergies and remind the user to check labels."
        )
        with st.spinner("AcneClear AI is preparing your meal ideas..."):
            try:
                st.session_state["acneclear_diet_response"] = ask_acneclear_ai(prompt)
            except Exception as exc:
                st.error(f"Could not reach the AI service: {type(exc).__name__}: {exc}")
    if st.session_state.get("acneclear_diet_response"):
        st.markdown("### Your AI-generated meal ideas")
        st.markdown(st.session_state["acneclear_diet_response"])
    st.info(
        "Meal ideas are general information, not a medical diet or treatment. "
        "For medical conditions, pregnancy, serious allergies, or medicine-related diet questions, "
        "consult a registered dietitian or clinician."
    )

elif page == "AI Chatbot":
    st.markdown(
        '<div class="eyebrow">ACNECLEAR AI ASSISTANT</div>'
        '<h1 class="hero-title">Ask me anything.</h1>'
        '<p class="hero-copy">Ask about acne education, skincare, recipes, studies, coding, technology, writing, science, or everyday questions.</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "One assistant for your entire AcneClear project and everyday questions. "
        "Ask about GFPGAN, Qwen3, LoRA, datasets, epochs, metrics, skincare, meals, coding, or general knowledge."
    )
    st.markdown(
        "**Try asking:** Explain my project architecture · What is the difference between GFPGAN and Qwen3? · "
        "How many epochs did I train? · What does LoRA do?"
    )
    if not _acneclear_setting("ACNECLEAR_CHAT_SPACE"):
        st.warning(
            "The chatbot backend is not connected yet. Upload your saved LoRA adapter to Hugging Face, "
            "deploy the chatbot_space folder as a Gradio Space, then add ACNECLEAR_CHAT_SPACE in Streamlit Secrets. "
            "Follow CHATBOT_DEPLOYMENT.md in the GitHub repository."
        )
    for item in st.session_state["acneclear_chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("Ask AcneClear AI a question...")
    if prompt:
        previous_history = st.session_state["acneclear_chat_history"][-8:]
        st.session_state["acneclear_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AcneClear AI is thinking..."):
                try:
                    reply = ask_acneclear_ai("[GENERAL TOPIC]\n" + prompt, previous_history)
                except Exception as exc:
                    reply = (
                        "The AI service is not connected or is temporarily unavailable. "
                        "Check the Hugging Face Space status and the ACNECLEAR_CHAT_SPACE secret. "
                        f"Technical detail: {type(exc).__name__}: {exc}"
                    )
                st.markdown(reply)
        st.session_state["acneclear_chat_history"].append({"role": "assistant", "content": reply})

    if st.button("Clear chat history"):
        st.session_state["acneclear_chat_history"] = []
        st.rerun()

st.markdown("---")
st.markdown('<div class="small-note">ACNECLEAR AI · Educational photo-retouching and wellness demo. Photo restoration is not a dedicated acne-removal model. Skincare and diet information is general education, not medical advice.</div>', unsafe_allow_html=True)