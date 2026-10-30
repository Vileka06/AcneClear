import base64
import io
import os
import requests
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
.brand-row {background:#fff;border:1px solid var(--line);border-radius:16px;padding:13px 17px;margin-bottom:16px;box-shadow:0 5px 20px rgba(65,42,35,.035);}
.intro-strip {background:linear-gradient(110deg,#fff 0%,#F8F3EA 100%);border:1px solid var(--line);border-radius:16px;padding:17px 20px;margin:0 0 22px 0;}
.intro-kicker {font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:#8C5966;font-weight:800;margin-bottom:5px;}
.intro-heading {font-family:'Manrope',sans-serif;font-size:21px;font-weight:800;color:var(--burgundy);line-height:1.2;}
.intro-copy {font-size:13px;color:var(--muted);line-height:1.55;margin-top:5px;}
.section-heading {background:#fff;border:1px solid var(--line);border-radius:15px;padding:18px 20px;margin-bottom:16px;}
.feature-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px 16px;min-height:115px;}
.feature-card h4 {margin:0 0 6px 0;color:var(--burgundy);font-family:'Manrope',sans-serif;font-size:15px;}
.feature-card p {margin:0;color:var(--muted);font-size:12px;line-height:1.5;}
.response-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:12px 0;color:var(--ink);}
[data-testid="stChatMessage"] {background:#fff;border:1px solid var(--line);border-radius:15px;padding:13px 15px;margin:10px 0;}
[data-testid="stChatInput"] {background:#fff;border:1px solid #CFC2B8;border-radius:14px;box-shadow:0 5px 18px rgba(65,42,35,.07);}
[data-testid="stChatInput"] textarea {background:transparent !important;color:#342B2B !important;}
.stChatFloatingInputContainer {background:rgba(243,235,221,.97) !important;border-top:1px solid var(--line);padding-top:10px;}
div[data-testid="stButton"] button {transition:all .15s ease;}
div[data-testid="stButton"] button:focus {border-color:var(--burgundy) !important;box-shadow:0 0 0 2px rgba(101,31,50,.12) !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-row">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="brand-mark">✳</div>
    <div>
      <div class="brand-name">AcneClear AI</div>
      <div class="brand-sub">Skin Wellness Studio · Photo retouching and everyday guidance</div>
    </div>
  </div>
  <div style="margin-left:auto"><span class="pill">AI WELLNESS STUDIO</span></div>
</div>
<div class="intro-strip">
  <div class="intro-kicker">YOUR PERSONAL SKIN WELLNESS WORKSPACE</div>
  <div class="intro-heading">Care for your skin. Make informed everyday choices.</div>
  <div class="intro-copy">Restore a portrait, explore gentle skincare guidance, plan balanced meals, or ask the AcneClear AI assistant.</div>
</div>
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


GEMINI_MODEL = "gemini-3.5-flash-lite"

ACNECLEAR_SYSTEM_PROMPT = """
You are AcneClear AI, a friendly general-purpose assistant and guide to the AcneClear AI student project.
Answer the user's actual question directly in simple, clear language. You may answer general questions about
education, coding, maths, science, writing and everyday topics.

VERIFIED ACNECLEAR PROJECT FACTS:
- AcneClear AI is a Streamlit website with four sections: Acne Remover, Beauty Tips, Diet Planner and AI Chatbot.
- Acne Remover: accepts JPG/PNG/WEBP portraits and uses pretrained TencentARC GFPGAN v1.3 (clean architecture,
  channel multiplier 2, upscale 2, restoration weight 0.6), followed by OpenCV bilateral smoothing and a YCrCb
  blending mask. It shows a before/after comparison and PNG download. GFPGAN is a general face-restoration model,
  NOT an acne-specific detector or medical acne remover. It was integrated as a pretrained model, not trained
  from scratch for this project.
- Beauty Tips: takes skin type, concerns and sensitivity, then generates general educational routine suggestions.
  It does not diagnose or prescribe.
- Diet Planner: takes food preferences, Indian meal style, budget, meals and foods to avoid, then generates
  general meal ideas. It is not a medical diet and does not claim food cures acne.
- LLM experiment: Qwen/Qwen3-0.6B, a pretrained decoder-only language model with 28 Transformer blocks,
  hidden size 1024, 16 attention heads, 8 key/value heads and intermediate size 3072.
- The Qwen3 experiment was fine-tuned separately with supervised fine-tuning and PEFT LoRA: rank 8,
  alpha 16, dropout 0.05. Target projections were q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj and down_proj.
  The saved adapter has 5,046,272 trainable parameters; the base model weights were frozen.
- The completed Colab experiment recorded 3 epochs, 62.47 minutes, 2,757 prepared examples, 2,205 train,
  276 validation and 276 test. Recorded test loss 0.7037, perplexity 2.0212, sampled response-token accuracy
  78.28% on 64 test examples, and final validation loss 0.7175. These are language-model diagnostics, not
  clinical accuracy or a guarantee that answers are factually correct.
- IMPORTANT DEPLOYMENT DISTINCTION: because the free Streamlit deployment cannot reliably host the trained
  Qwen adapter as an online inference server without another compute service, this no-cost website chatbot
  uses the Gemini Developer API free tier. The Qwen3 LoRA adapter remains the user's separate completed
  fine-tuning experiment; do not claim that Gemini is the fine-tuned Qwen model.
- The live Gemini model selected for the website is gemini-3.5-flash-lite. This assistant has no live web search
  unless a separate search service is added.

SAFETY:
For acne and skincare, give general education only. Do not diagnose, prescribe medicines, suggest changing
prescribed treatment or promise a cure. For deep, painful, persistent or scarring acne, recommend a dermatologist.
For food allergies, pregnancy, medical conditions or medicine-related dietary questions, recommend a qualified
clinician or registered dietitian. Do not invent nutrition values or create crash diets. If uncertain, say so.
"""

def ask_acneclear_ai(prompt: str, history=None) -> str:
    """Call Gemini's free-tier API from the Streamlit server; the key never reaches the browser."""
    api_key = _acneclear_setting("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Add a free Gemini API key as GEMINI_API_KEY in Streamlit Community Cloud → App settings → Secrets. "
            "Create a key at https://aistudio.google.com/apikey. Do not enable paid billing."
        )

    contents = []
    for item in (history or [])[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("content")
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": text[:1800]}],
            })

    contents.append({
        "role": "user",
        "parts": [{"text": str(prompt).strip()[:4000]}],
    })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": ACNECLEAR_SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 750},
    }

    try:
        response = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 60),
        )
    except requests.Timeout as exc:
        raise RuntimeError("The free AI service timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Could not reach the AI service. Check the internet connection and try again.") from exc

    if response.status_code == 429:
        raise RuntimeError(
            "The Gemini free-tier quota or rate limit has been reached. Wait for the quota to reset, "
            "or check the limits in Google AI Studio. No paid billing is required for this setup."
        )
    if response.status_code == 403:
        raise RuntimeError(
            "Google AI Studio did not allow this request. Check that the API key is valid and that the project "
            "is eligible for Gemini's free tier. Do not enable billing if you want to keep this project free."
        )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get("message", "Unknown API error")
        except (ValueError, AttributeError):
            detail = response.text[:300]
        raise RuntimeError(f"Gemini API error ({response.status_code}): {detail}")

    try:
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        answer = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The AI service returned an empty or unexpected response. Please rephrase your question.") from exc

    answer = answer.strip()
    if not answer:
        raise RuntimeError("The AI service did not return text. Please rephrase your question.")
    return answer


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
        '<p class="hero-copy">Tell AcneClear AI a little about your skin and ask a skincare question. The AI assistant will create an educational routine suggestion.</p>',
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
        st.markdown('<div class="response-card">', unsafe_allow_html=True)
        st.markdown(st.session_state["acneclear_beauty_response"])
        st.markdown('</div>', unsafe_allow_html=True)
    st.info(
        "This is general skincare education, not a diagnosis or treatment plan. "
        "For painful, persistent, worsening, or scarring acne, consult a dermatologist."
    )

elif page == "Diet Planner":
    st.markdown(
        '<div class="eyebrow">THE EVERYDAY FOOD GUIDE</div>'
        '<h1 class="hero-title">Meal ideas made around your preferences.</h1>'
        '<p class="hero-copy">Choose your food preference and ask the AcneClear AI assistant for balanced, practical Indian meal ideas.</p>',
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
        st.markdown('<div class="response-card">', unsafe_allow_html=True)
        st.markdown(st.session_state["acneclear_diet_response"])
        st.markdown('</div>', unsafe_allow_html=True)
    st.info(
        "Meal ideas are general information, not a medical diet or treatment. "
        "For medical conditions, pregnancy, serious allergies, or medicine-related diet questions, "
        "consult a registered dietitian or clinician."
    )

elif page == "AI Chatbot":
    st.markdown(
        '<div class="section-heading">'
        '<div class="eyebrow">ACNECLEAR AI ASSISTANT</div>'
        '<h1 class="hero-title" style="font-size:34px">Ask anything. Understand your project.</h1>'
        '<p class="hero-copy">One assistant for everyday questions and everything behind AcneClear AI: GFPGAN, Qwen3, LoRA, datasets, training, skincare and meal planning.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not _acneclear_setting("ACNECLEAR_CHAT_SPACE"):
        st.warning(
            "Your chatbot model is not connected yet. Upload the trained adapter, deploy the chatbot_space folder as a Hugging Face Gradio Space, "
            "then add ACNECLEAR_CHAT_SPACE in Streamlit Secrets. Follow CHATBOT_DEPLOYMENT.md in GitHub."
        )

    pending_prompt = st.session_state.pop("acneclear_pending_prompt", None)

    if not st.session_state["acneclear_chat_history"] and not pending_prompt:
        st.markdown(
            '<div class="response-card">'
            '<h3 style="color:#651F32;margin-top:0">Welcome to AcneClear AI</h3>'
            '<p style="color:#756B68">I can explain how your project works, help with skincare and meal ideas, or answer general questions. '
            'Choose a suggested question or type your own below.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Explore a topic**")
        prompt_options = [
            ("Project overview", "Explain my complete AcneClear AI project architecture in simple words."),
            ("My trained LLM", "Explain Qwen3-0.6B, its 28 transformer blocks and how I fine-tuned it using LoRA."),
            ("Training results", "Explain my 3 training epochs, dataset split, test loss, perplexity and token accuracy."),
            ("GFPGAN vs LLM", "What is the difference between GFPGAN and Qwen3 in my project?"),
        ]
        col_a, col_b = st.columns(2, gap="medium")
        for index, (label, prompt_text) in enumerate(prompt_options):
            target = col_a if index % 2 == 0 else col_b
            with target:
                if st.button(label, key=f"chat_suggestion_{index}", use_container_width=True):
                    st.session_state["acneclear_pending_prompt"] = prompt_text
                    st.rerun()

    for item in st.session_state["acneclear_chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = pending_prompt or st.chat_input("Message AcneClear AI...")
    if prompt:
        previous_history = st.session_state["acneclear_chat_history"][-8:]
        st.session_state["acneclear_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AcneClear AI is thinking..."):
                try:
                    reply = ask_acneclear_ai(prompt, previous_history)
                except Exception as exc:
                    reply = (
                        "I could not connect to the AcneClear AI model service. "
                        "Please check that GEMINI_API_KEY is set in Streamlit Secrets and that your Gemini free-tier quota is available. "
                        f"Technical detail: {type(exc).__name__}: {exc}"
                    )
                st.markdown(reply)
        st.session_state["acneclear_chat_history"].append({"role": "assistant", "content": reply})

    if st.session_state["acneclear_chat_history"]:
        if st.button("Start a new conversation", key="clear_chat_history"):
            st.session_state["acneclear_chat_history"] = []
            st.rerun()

st.markdown("---")
st.markdown('<div class="small-note">ACNECLEAR AI · Educational photo-retouching and wellness demo. Photo restoration is not a dedicated acne-removal model. Skincare and diet information is general education, not medical advice.</div>', unsafe_allow_html=True)