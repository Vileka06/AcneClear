from io import BytesIO
from pathlib import Path
import hashlib
import os
import sys
import urllib.request

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------------------------------------------------
# AcneClear AI | Acne retouching + skincare chatbot
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
WEIGHT_PATH = MODEL_DIR / "GFPGANv1.3.pth"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(
    page_title="AcneClear AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep only the two requested features.
for key, default in {
    "acneclear_original": None,
    "acneclear_result": None,
    "acneclear_photo_hash": None,
    "acneclear_chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --navy:#243B63;
  --navy-deep:#172B4D;
  --blue:#EAF1F8;
  --paper:#F7F9FC;
  --line:#DCE5EF;
  --muted:#68788D;
  --coral:#D98270;
}
.stApp { background:var(--paper); color:var(--navy-deep); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1160px; padding-top:1.3rem; padding-bottom:3rem; }
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] { background:rgba(247,249,252,.96); }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; color:var(--navy) !important; }
.brandbar {
  display:flex; align-items:center; gap:13px; padding:15px 18px;
  background:#fff; border:1px solid var(--line); border-radius:16px;
  box-shadow:0 5px 20px rgba(36,59,99,.04); margin-bottom:22px;
}
.brand-icon {
  width:43px;height:43px;border-radius:12px;background:var(--navy);
  color:white;display:flex;align-items:center;justify-content:center;
  font:800 21px Manrope;
}
.brand-title {font:800 20px Manrope;color:var(--navy-deep);line-height:1.15;}
.brand-sub {font-size:10px;letter-spacing:1.6px;color:var(--muted);margin-top:4px;}
.demo-link {margin-left:auto;font-size:12px;font-weight:700;color:var(--navy);text-decoration:none;}
.hero {
  background:linear-gradient(115deg,#fff 0%,#F0F5FA 100%);
  border:1px solid var(--line); border-radius:18px; padding:27px 30px;
  margin-bottom:18px; position:relative; overflow:hidden;
}
.hero:after {
  content:""; position:absolute; width:190px;height:190px;border-radius:50%;
  right:-65px;top:-95px;border:28px solid rgba(36,59,99,.045);
}
.kicker {font-size:10px;font-weight:800;letter-spacing:2px;color:#6680A0;text-transform:uppercase;margin-bottom:8px;}
.hero h1 {font-size:clamp(27px,3.5vw,39px);line-height:1.13;letter-spacing:-1px;margin:0;}
.hero p {color:var(--muted);font-size:14px;line-height:1.7;max-width:760px;margin:10px 0 0;}
div[data-baseweb="tab-list"] {gap:8px;border-bottom:1px solid var(--line);padding-bottom:8px;}
button[data-baseweb="tab"] {
  background:#fff;border:1px solid var(--line);border-radius:9px 9px 9px 9px;
  color:var(--navy);font-weight:700;padding:10px 18px;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background:var(--navy);color:#fff;border-color:var(--navy);
}
div.stButton>button,div.stDownloadButton>button {
  background:var(--navy);color:#fff;border:1px solid var(--navy);
  border-radius:9px;min-height:43px;font-weight:700;
}
div.stButton>button:hover,div.stDownloadButton>button:hover {
  background:#34527D;color:#fff;border-color:#34527D;
}
div.stButton>button p,div.stDownloadButton>button p {color:#fff;}
div[data-testid="stFileUploader"] {
  background:#fff;border:1px dashed #AFC1D5;border-radius:13px;padding:12px;
}
div[data-testid="stImage"] img {border-radius:13px;border:1px solid var(--line);}
div[data-testid="stChatMessage"] {
  background:#fff;border:1px solid var(--line);border-radius:13px;padding:12px 15px;
}
[data-testid="stChatInput"] {
  background:#fff;border:1px solid var(--line);border-radius:13px;
}
[data-testid="stChatInput"] textarea {background:#fff !important;color:var(--navy-deep) !important;}
.stChatFloatingInputContainer {background:rgba(247,249,252,.97) !important;}

/* Force readable contrast in chat bubbles and the floating chat composer. */
div[data-testid="stChatMessage"],
div[data-testid="stChatMessage"] *,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em {
  color: #172B4D !important;
  opacity: 1 !important;
}
div[data-testid="stChatMessage"] a { color: #244F86 !important; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder {
  color: #172B4D !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #172B4D !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #68788D !important;
  -webkit-text-fill-color: #68788D !important;
  opacity: 1 !important;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  color: #172B4D;
}

.small-note {font-size:11px;color:var(--muted);line-height:1.6;}
.section-card {
  background:#fff;border:1px solid var(--line);border-radius:15px;
  padding:18px 20px;margin:8px 0 15px;
}
.step {font-size:10px;font-weight:800;letter-spacing:1.5px;color:#7890AA;text-transform:uppercase;}
.section-title {font:700 19px Manrope;color:var(--navy);margin:4px 0;}
.section-copy {font-size:13px;color:var(--muted);line-height:1.6;}
.status {display:inline-block;background:#EAF1F8;color:#34527D;border-radius:6px;padding:6px 9px;font-size:11px;font-weight:700;}
hr {border-color:var(--line);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brandbar">
  <div class="brand-icon">AC</div>
  <div>
    <div class="brand-title">AcneClear AI</div>
    <div class="brand-sub">SKIN RETOUCHING · AI SKINCARE CHAT</div>
  </div>
  <a class="demo-link" href="https://acneclear.streamlit.app/" target="_blank">LIVE DEMO ↗</a>
</div>
<div class="hero">
  <div class="kicker">A simpler way to care for your skin</div>
  <h1>Clearer-looking photos.<br>Smarter skincare conversations.</h1>
  <p>Retouch a portrait with the Acne Remover, or chat with the AcneClear AI assistant for simple, educational skincare guidance.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_gfpgan():
    # Compatibility shim for BasicSR versions expecting this older torchvision module.
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from gfpgan import GFPGANer

    if not WEIGHT_PATH.exists():
        urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)

    return GFPGANer(
        model_path=str(WEIGHT_PATH),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def soften_skin(image_rgb: np.ndarray, strength: int = 35) -> np.ndarray:
    """Apply gentle texture smoothing; this is not a dedicated acne-segmentation model."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(
        bgr,
        d=9,
        sigmaColor=30 + strength * 1.25,
        sigmaSpace=30 + strength * 1.25,
    )
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array([0, 128, 70], dtype=np.uint8),
        np.array([255, 180, 145], dtype=np.uint8),
    )
    mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32) / 255.0
    mask = (mask * (strength / 100.0) * 0.72)[:, :, None]
    blended = bgr.astype(np.float32) * (1.0 - mask) + smooth.astype(np.float32) * mask
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def retouch_photo(photo: Image.Image) -> Image.Image:
    rgb = np.asarray(photo.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restorer = load_gfpgan()
    _, _, restored_bgr = restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.6,
    )
    if restored_bgr is None:
        raise RuntimeError("No face could be restored. Try a clear, front-facing portrait.")
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(soften_skin(restored_rgb, 35))


def get_secret(name: str) -> str:
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    return str(secret_value or os.getenv(name, "")).strip()


CHAT_SYSTEM_PROMPT = """
You are AcneClear AI, a friendly assistant focused on acne, skincare and everyday skin wellness.
Answer the user's actual question in clear, practical, non-judgmental language. Keep answers concise and
explain unfamiliar terms simply. You may discuss gentle skincare routines, common acne triggers, product
ingredient education, sunscreen, skin hygiene, and when to seek professional help.

SAFETY:
- Provide general educational information, not a medical diagnosis or personalised medical treatment.
- Do not prescribe medicines, change a user's prescribed treatment, or promise that a product or food will cure acne.
- Do not claim to identify a skin disease from a photo.
- Encourage a dermatologist for deep, painful, persistent, worsening or scarring acne.
- For pregnancy, allergies, medical conditions or medication questions, recommend a qualified clinician.
- Do not shame the user about their appearance. Be supportive and realistic.
"""


def ask_gemini_stream(prompt: str, history: list):
    """Stream a short Gemini answer so the user sees text as it is generated."""
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "The chatbot API key is not configured. Add GEMINI_API_KEY in "
            "Streamlit Community Cloud → App settings → Secrets."
        )

    # A short recent context reduces tokens sent on every turn.
    contents = []
    for item in history[-4:]:
        role = item.get("role")
        message = item.get("content", "")
        if role in ("user", "assistant") and isinstance(message, str) and message.strip():
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": message[:900]}],
            })
    contents.append({"role": "user", "parts": [{"text": prompt[:2000]}]})

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:streamGenerateContent?alt=sse"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": CHAT_SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 320},
    }

    try:
        response = requests.post(
            endpoint,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(8, 35),
            stream=True,
        )
    except requests.Timeout as exc:
        raise RuntimeError("Gemini took too long to start. Please try again.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Could not connect to Gemini. Please try again.") from exc

    if response.status_code == 429:
        response.close()
        raise RuntimeError("The Gemini request limit has been reached. Please wait and try again.")
    if response.status_code == 503:
        response.close()
        raise RuntimeError("Gemini is temporarily busy. Please try again in a moment.")
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get("message", "Check the API key and model access.")
        except (ValueError, AttributeError):
            detail = "Check the API key and model access."
        status = response.status_code
        response.close()
        raise RuntimeError(f"Gemini returned error {status}: {detail}")

    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = __import__("json").loads(chunk)
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for part in parts:
                    text_part = part.get("text")
                    if text_part:
                        yield text_part
            except (ValueError, IndexError, TypeError, AttributeError):
                continue
    except requests.RequestException as exc:
        raise RuntimeError("The connection to Gemini was interrupted. Please try again.") from exc
    finally:
        response.close()


tab_remover, tab_chat = st.tabs(["✦  Acne Remover", "☏  AI Chatbot"])

with tab_remover:
    st.markdown("""
    <div class="section-card">
      <div class="step">01 · PHOTO RETOUCHING</div>
      <div class="section-title">Acne Remover</div>
      <div class="section-copy">Upload a clear portrait to create a softly retouched preview. Compare the original and result, then save your image.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your portrait",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG or WEBP · Maximum 10 MB",
        key="acneclear_upload",
    )

    if uploaded is not None:
        raw = uploaded.getvalue()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("Please upload an image smaller than 10 MB.")
        else:
            try:
                photo_hash = hashlib.sha256(raw).hexdigest()
                original = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
                if st.session_state["acneclear_photo_hash"] != photo_hash:
                    st.session_state["acneclear_photo_hash"] = photo_hash
                    st.session_state["acneclear_original"] = original
                    st.session_state["acneclear_result"] = None

                st.markdown('<div class="step">02 · YOUR PHOTO</div>', unsafe_allow_html=True)
                st.image(original, caption="Original photo", use_container_width=True)

                if st.button("Retouch my photo", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Retouching your photo. The first run may take longer while GFPGAN loads..."):
                            result = retouch_photo(original)
                        st.session_state["acneclear_result"] = result
                        st.success("Your retouched preview is ready.")
                    except Exception as exc:
                        st.session_state["acneclear_result"] = None
                        st.error("The photo could not be retouched. Try a clear, front-facing portrait.")
                        st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

                result = st.session_state.get("acneclear_result")
                if result is not None and st.session_state.get("acneclear_photo_hash") == photo_hash:
                    st.markdown('<div class="step">03 · BEFORE & AFTER</div>', unsafe_allow_html=True)
                    before_col, after_col = st.columns(2, gap="large")
                    with before_col:
                        st.image(original, caption="Before", use_container_width=True)
                    with after_col:
                        st.image(result, caption="After", use_container_width=True)

                    output = BytesIO()
                    result.save(output, format="PNG")
                    st.download_button(
                        "Download retouched photo",
                        data=output.getvalue(),
                        file_name="acneclear_retouched.png",
                        mime="image/png",
                        use_container_width=True,
                    )
            except (UnidentifiedImageError, OSError):
                st.error("This file could not be opened. Please choose another JPG, PNG or WEBP image.")
    else:
        st.session_state["acneclear_photo_hash"] = None
        st.session_state["acneclear_original"] = None
        st.session_state["acneclear_result"] = None

    st.markdown(
        '<p class="small-note">Photo retouching is cosmetic. GFPGAN restores facial details and the app applies gentle texture smoothing; it is not a dedicated acne-detection or medical treatment model. Results vary by photo.</p>',
        unsafe_allow_html=True,
    )


with tab_chat:
    st.markdown("""
    <div class="section-card">
      <div class="step">02 · SKINCARE CONVERSATION</div>
      <div class="section-title">Ask AcneClear AI</div>
      <div class="section-copy">Ask about acne, simple skincare routines, ingredients, sunscreen and everyday skin wellness.</div>
    </div>
    """, unsafe_allow_html=True)

    api_ready = bool(get_secret("GEMINI_API_KEY"))
    if api_ready:
        st.markdown('<span class="status">● AI assistant connected</span>', unsafe_allow_html=True)
    else:
        st.warning(
            "To enable the chatbot, add GEMINI_API_KEY in Streamlit Community Cloud → "
            "App settings → Secrets. Never add the key to GitHub."
        )

    top_left, top_right = st.columns([0.75, 0.25])
    with top_left:
        st.caption("Your conversation stays in this app session.")
    with top_right:
        if st.button("New chat", use_container_width=True):
            st.session_state["acneclear_chat_history"] = []
            st.rerun()

    with st.container(height=470, border=True):
        if not st.session_state["acneclear_chat_history"]:
            st.markdown(
                "<div style='padding:34px 12px;text-align:center'>"
                "<div style='font-size:11px;letter-spacing:1.7px;font-weight:800;color:#7890AA'>WELCOME TO ACNECLEAR</div>"
                "<h3 style='margin:10px 0 6px'>How can I help your skin today?</h3>"
                "<p style='color:#68788D;font-size:13px'>Ask a question about acne or skincare to get started.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        for message in st.session_state["acneclear_chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about acne or skincare...",
        disabled=not api_ready,
    )
    if prompt:
        previous = st.session_state["acneclear_chat_history"][-8:]
        st.session_state["acneclear_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AcneClear AI is thinking..."):
                try:
                    answer = st.write_stream(ask_gemini_stream(prompt, previous))
                    if not answer:
                        answer = "I couldn't get a response from Gemini. Please try again."
                        st.markdown(answer)
                except Exception as exc:
                    answer = f"I couldn't get a response just now. {exc}"
                    st.markdown(answer)
        st.session_state["acneclear_chat_history"].append(
            {"role": "assistant", "content": answer}
        )

st.markdown(
    '<hr><p class="small-note" style="text-align:center">ACNECLEAR AI · Cosmetic photo retouching and educational skincare chat. Not a medical diagnosis or treatment service.</p>',
    unsafe_allow_html=True,
)
