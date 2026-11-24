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
/* Force readable foreground colours across Streamlit's themed widgets. */
.stApp, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stText"],
.stApp [data-testid="stChatMessage"] p,
.stApp [data-testid="stChatMessage"] li,
.stApp [data-testid="stChatMessage"] div,
.stApp [data-testid="stFileUploader"] small,
.stApp [data-testid="stFileUploader"] span {
  color:var(--navy-deep) !important;
}
.stApp input, .stApp textarea, .stApp [contenteditable="true"] {
  color:var(--navy-deep) !important;
  -webkit-text-fill-color:var(--navy-deep) !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
  color:#68788D !important;
  -webkit-text-fill-color:#68788D !important;
  opacity:1 !important;
}
.stApp [data-testid="stChatMessage"] {
  color:var(--navy-deep) !important;
}

.block-container { max-width:1160px; padding-top:1.3rem; padding-bottom:3rem; }
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] { background:rgba(247,249,252,.96); }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; color:var(--navy) !important; }
.brandbar {
  display:flex; align-items:center; gap:14px; padding:13px 18px;
  background:#fff; border:1px solid var(--line); border-radius:18px;
  box-shadow:0 8px 28px rgba(36,59,99,.06); margin-bottom:20px;
}
.brand-icon {
  width:46px;height:46px;border-radius:15px;background:linear-gradient(145deg,#243B63,#54749D);
  color:white;display:flex;align-items:center;justify-content:center;
  font:800 20px Manrope; box-shadow:0 6px 14px rgba(36,59,99,.2);
}
.brand-title {font:800 20px Manrope;color:var(--navy-deep);line-height:1.15;}
.brand-sub {font-size:10px;letter-spacing:1.5px;color:var(--muted);margin-top:4px;}
.demo-link {margin-left:auto;font-size:12px;font-weight:700;color:var(--navy);text-decoration:none;background:#EAF1F8;padding:10px 15px;border-radius:20px;}
.hero {
  min-height:300px; display:flex; flex-direction:column; justify-content:center;
  background:radial-gradient(ellipse at 82% 25%,rgba(255,255,255,.88) 0 9%,transparent 10%),
             radial-gradient(ellipse at 88% 65%,rgba(184,207,231,.65) 0 18%,transparent 19%),
             linear-gradient(112deg,#DCE8F5 0%,#B9CDE3 48%,#EAF1F8 100%);
  border:1px solid #D5E1EE; border-radius:26px; padding:38px 42px;
  margin-bottom:22px; position:relative; overflow:hidden;
  box-shadow:0 16px 36px rgba(36,59,99,.08);
}
.hero:after {
  content:""; position:absolute; width:260px;height:260px;border-radius:48% 52% 60% 40%;
  right:7%;top:15px;transform:rotate(-18deg);
  background:linear-gradient(145deg,rgba(255,255,255,.75),rgba(106,139,177,.18));
  border:1px solid rgba(255,255,255,.8); box-shadow:inset 0 0 35px rgba(255,255,255,.45);
}
.hero:before {
  content:""; position:absolute; width:190px;height:190px;border-radius:50%;
  right:12%;top:48px;border:1px solid rgba(36,59,99,.13);
  box-shadow:0 0 0 18px rgba(255,255,255,.13),0 0 0 42px rgba(255,255,255,.1);
}
.hero-content {position:relative;z-index:2;max-width:60%;}
.kicker {font-size:10px;font-weight:800;letter-spacing:2px;color:#496789;text-transform:uppercase;margin-bottom:10px;}
.hero h1 {font-size:clamp(30px,4vw,48px);line-height:1.08;letter-spacing:-1.5px;margin:0;color:#172B4D !important;}
.hero p {color:#405875;font-size:14px;line-height:1.7;max-width:500px;margin:13px 0 0;}
.hero-chip {display:inline-block;margin-top:20px;background:rgba(255,255,255,.82);border:1px solid rgba(255,255,255,.9);padding:9px 14px;border-radius:20px;color:#243B63;font-size:11px;font-weight:700;}
@media(max-width:700px){.hero{padding:28px 22px;min-height:270px}.hero-content{max-width:100%}.hero:before,.hero:after{opacity:.28;right:-55px}.brand-sub{letter-spacing:.7px}.demo-link{padding:8px 10px;font-size:10px;}}
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
  <div class="hero-content">
    <div class="kicker">ACNECLEAR AI · YOUR PERSONAL SKINCARE SPACE</div>
    <h1>Feel confident.<br>Love your natural glow.</h1>
    <p>Enhance your portrait and get simple answers to your skincare questions, all in one calm and easy-to-use space.</p>
    <span class="hero-chip">Photo enhancement&nbsp;&nbsp; · &nbsp;&nbsp;AI skincare chat</span>
  </div>
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


def ask_gemini(prompt: str, history: list) -> str:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "The chatbot API key is not configured. Add GEMINI_API_KEY in "
            "Streamlit Community Cloud → App settings → Secrets."
        )

    contents = []
    for item in history[-8:]:
        role = item.get("role")
        message = item.get("content", "")
        if role in ("user", "assistant") and isinstance(message, str) and message.strip():
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": [{"text": message[:1800]}],
            })
    contents.append({"role": "user", "parts": [{"text": prompt[:4000]}]})

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": CHAT_SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 700},
    }

    try:
        response = requests.post(
            endpoint,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 60),
        )
    except requests.Timeout as exc:
        raise RuntimeError("The AI service took too long to respond. Please try again.") from exc
    except requests.RequestException as exc:
        raise RuntimeError("Could not connect to the AI service. Please try again.") from exc

    if response.status_code == 429:
        raise RuntimeError("The Gemini request limit has been reached. Wait a little and try again.")
    if response.status_code == 503:
        raise RuntimeError("Gemini is temporarily busy (503). Please wait a moment and send your message again.")
    if response.status_code in (400, 403, 404):
        try:
            detail = response.json().get("error", {}).get("message", "Check the API key and model access.")
        except (ValueError, AttributeError):
            detail = "Check the API key and model access."
        raise RuntimeError(f"Gemini could not process the request ({response.status_code}): {detail}")
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini returned an error ({response.status_code}). Please try again.")

    try:
        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        answer = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The AI returned an empty response. Please rephrase your question.") from exc

    if not answer:
        raise RuntimeError("The AI returned an empty response. Please try again.")
    return answer


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
                    answer = ask_gemini(prompt, previous)
                except Exception as exc:
                    answer = (
                        "I couldn't get a response just now. "
                        f"{exc}"
                    )
                st.markdown(answer)
        st.session_state["acneclear_chat_history"].append(
            {"role": "assistant", "content": answer}
        )

st.markdown(
    '<hr><p class="small-note" style="text-align:center">ACNECLEAR AI · Cosmetic photo retouching and educational skincare chat. Not a medical diagnosis or treatment service.</p>',
    unsafe_allow_html=True,
)
