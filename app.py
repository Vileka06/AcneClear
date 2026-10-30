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

# Palette: white, muted blue-grey, deep navy
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --cream:#F7F9FC; --sage:#B8C7D9; --burgundy:#17365D;
  --ink:#172B4D; --muted:#52657A; --line:#D8E1EC; --white:#FFFFFF;
}
.stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1320px; padding-top:1.3rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:rgba(247,249,252,.98); }
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
.pill {display:inline-block;border:1px solid var(--sage);background:#EAF0F7;color:var(--burgundy);border-radius:6px;padding:5px 9px;font-size:11px;font-weight:700;}
div.stButton > button, div.stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
 background:var(--burgundy);color:#fff;border:1px solid var(--burgundy);border-radius:8px;min-height:42px;font-weight:700;padding:.55rem 1rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {background:#244C7A !important;border-color:#244C7A !important;color:#fff !important;}
div.stButton > button p, div.stDownloadButton > button p, div[data-testid="stFormSubmitButton"] > button p {color:#fff !important;}
div[data-baseweb="input"] input, div[data-baseweb="select"] input, div[data-baseweb="textarea"] textarea {background:#fff !important;color:#172B4D !important;}
div[data-baseweb="select"] > div {background:#fff !important;color:#172B4D !important;}
div[data-testid="stImage"] img {border-radius:14px;}
div[data-testid="stFileUploader"] {background:#fff;border:1px dashed var(--sage);border-radius:12px;padding:10px;}
[data-testid="stSidebar"] {background:#F3F6FA;border-right:1px solid var(--line);}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:var(--burgundy);font-family:'Manrope',sans-serif;}
[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;}
div[data-baseweb="tab-list"] {gap:8px;}
button[data-baseweb="tab"] {background:#EAF0F7;border-radius:8px 8px 0 0;color:var(--burgundy);padding:10px 16px;}
button[data-baseweb="tab"][aria-selected="true"] {background:var(--burgundy);color:#fff;}
.stSlider [data-baseweb="slider"] div {background:var(--burgundy);}
[data-testid="stChatInput"] textarea {background:#fff !important;color:#172B4D !important;}
[data-testid="stChatInput"] textarea::placeholder {color:#52657A !important;opacity:1 !important;}
[data-testid="stChatInput"] button {color:#17365D !important;}
.small-note {color:var(--muted);font-size:11px;line-height:1.6;}
hr {border-color:var(--line);}
.brand-row {background:#fff;border:1px solid var(--line);border-radius:16px;padding:13px 17px;margin-bottom:16px;box-shadow:0 5px 20px rgba(65,42,35,.035);}
.intro-strip {background:linear-gradient(110deg,#fff 0%,#F3F6FA 100%);border:1px solid var(--line);border-radius:16px;padding:17px 20px;margin:0 0 22px 0;}
.intro-kicker {font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:#496A91;font-weight:800;margin-bottom:5px;}
.intro-heading {font-family:'Manrope',sans-serif;font-size:21px;font-weight:800;color:var(--burgundy);line-height:1.2;}
.intro-copy {font-size:13px;color:var(--muted);line-height:1.55;margin-top:5px;}
.section-heading {background:#fff;border:1px solid var(--line);border-radius:15px;padding:18px 20px;margin-bottom:16px;}
.feature-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px 16px;min-height:115px;}
.feature-card h4 {margin:0 0 6px 0;color:var(--burgundy);font-family:'Manrope',sans-serif;font-size:15px;}
.feature-card p {margin:0;color:var(--muted);font-size:12px;line-height:1.5;}
.response-card {background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:12px 0;color:var(--ink);}
[data-testid="stChatMessage"] {background:#fff;border:1px solid var(--line);border-radius:15px;padding:13px 15px;margin:10px 0;}
[data-testid="stChatInput"] {background:#fff;border:1px solid #D8E1EC;border-radius:14px;box-shadow:0 5px 18px rgba(65,42,35,.07);}
[data-testid="stChatInput"] textarea {background:transparent !important;color:#172B4D !important;}
.stChatFloatingInputContainer {background:rgba(247,249,252,.98) !important;border-top:1px solid var(--line);padding-top:10px;}
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
*{{box-sizing:border-box}}body{{margin:0;background:#F7F9FC;font-family:Arial,sans-serif}}
.compare{{position:relative;width:100%;height:{height}px;overflow:hidden;border-radius:12px;background:#fff;border:1px solid #D8E1EC;--pos:50%}}
.compare img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff}}
.after{{clip-path:inset(0 0 0 var(--pos))}}
.line{{position:absolute;top:0;bottom:0;left:var(--pos);width:2px;background:#fff;box-shadow:0 0 0 1px #17365D40;pointer-events:none}}
.knob{{position:absolute;left:var(--pos);top:50%;transform:translate(-50%,-50%);width:38px;height:38px;border-radius:50%;background:#17365D;border:2px solid #fff;box-shadow:0 2px 10px #0002;display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px;pointer-events:none}}
input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
.label{{position:absolute;top:12px;padding:7px 10px;border-radius:5px;background:#17365D;color:#fff;font-size:11px;font-weight:700;letter-spacing:.3px}}
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


def offline_beauty_starter(skin_type: str, concerns: list, sensitivity: str, question: str) -> str:
    """Safe, general fallback when the free AI API key has not been configured."""
    if skin_type == "Oily":
        moisturizer = "Choose a light, oil-free or non-comedogenic moisturizer if it suits your skin."
    elif skin_type == "Dry":
        moisturizer = "Choose a gentle, fragrance-free cream moisturizer that feels comfortable."
    elif skin_type == "Sensitive" or "Sensitive" in sensitivity:
        moisturizer = "Choose a fragrance-free moisturizer and stop any product that causes persistent stinging."
    elif skin_type == "Combination":
        moisturizer = "Use a light moisturizer overall and add a little more to dry areas if needed."
    else:
        moisturizer = "Use a simple moisturizer that feels comfortable and does not irritate your skin."

    concern_text = ", ".join(concerns) if concerns else "general skin care"
    question_line = (
        f"\n\n**Your question:** {question}\n\n"
        "This offline starter cannot provide a personalized answer to the extra question. "
        "Connect the AI key to get a generated response."
        if question.strip() else ""
    )
    return f"""### Your starter routine
**Skin type:** {skin_type}  
**Concerns selected:** {concern_text}

#### Morning
1. Gently wash with a mild, non-abrasive cleanser or rinse with lukewarm water.
2. {moisturizer}
3. Apply a broad-spectrum SPF 30+ sunscreen. Follow the product label and reapply as directed.

#### Evening
1. Gently remove makeup and sunscreen.
2. Wash with a mild cleanser. Do not scrub or use rough brushes.
3. Apply your moisturizer.

#### Habits to avoid
- Do not squeeze or pick pimples.
- Avoid harsh scrubs, lemon juice, toothpaste and several new products at once.
- Introduce only one new skincare product at a time.

#### When to get professional advice
Painful, deep, persistent or scarring acne should be assessed by a dermatologist. This routine is general education, not an acne treatment or a guarantee of clear skin.{question_line}"""


def offline_diet_starter(
    preference: str,
    meal_style: str,
    meals_per_day: str,
    foods_to_avoid: str,
    budget: str,
    additional_request: str,
) -> str:
    """Conservative Indian meal ideas for demo mode; not a medical diet."""
    avoid_text = foods_to_avoid.strip()
    allergy_terms = (
        "allerg", "peanut", "groundnut", "tree nut", "cashew", "almond",
        "milk allergy", "dairy allergy", "lactose allergy", "gluten allergy",
        "wheat allergy", "soy allergy", "soya allergy", "sesame allergy",
        "shellfish", "seafood allergy", "fish allergy", "egg allergy"
    )
    if avoid_text and any(term in avoid_text.lower() for term in allergy_terms):
        return (
            "### Allergy-safe planning needs a verified ingredient check\n"
            f"You listed: **{avoid_text}**.\n\n"
            "I have not generated an offline menu because a template cannot verify every ingredient, packaged-food label, "
            "or kitchen cross-contact risk. Please check every ingredient with the food provider and follow your clinician's allergy plan. "
            "For a serious or multiple-food allergy, consult a registered dietitian."
        )

    vegan = preference == "Vegan"
    eggs = preference == "Includes eggs"
    meat = preference == "Includes meat or fish"
    south = meal_style == "South Indian"
    north = meal_style == "North Indian"

    if south:
        if vegan:
            breakfast = "Idli with vegetable sambar and coconut chutney; check that the sambar and chutney contain no ghee or dairy."
            lunch = "Rice with sambar, vegetable poriyal and chickpea sundal."
            dinner = "Ragi dosa with vegetable sambar and a seasonal vegetable side."
        elif eggs:
            breakfast = "Idli with sambar and one well-cooked boiled egg."
            lunch = "Rice with sambar, vegetable poriyal and well-cooked egg curry."
            dinner = "Dosa with sambar and a side of dal."
        elif meat:
            breakfast = "Idli with sambar and a seasonal fruit."
            lunch = "Rice with vegetable sambar, poriyal and well-cooked fish or chicken."
            dinner = "Dosa with sambar and a vegetable side."
        else:
            breakfast = "Idli with sambar and coconut chutney."
            lunch = "Rice with sambar, vegetable poriyal and curd if tolerated."
            dinner = "Ragi dosa with sambar and a seasonal vegetable side."
        snack_options = ["A seasonal fruit such as banana or guava", "Roasted chana", "Steamed sundal"]
    elif north:
        if vegan:
            breakfast = "Vegetable poha with lemon and seasonal fruit."
            lunch = "Chapati with chana masala and seasonal sabzi."
            dinner = "Moong dal khichdi with cucumber salad."
        elif eggs:
            breakfast = "Vegetable poha with one well-cooked egg."
            lunch = "Chapati with dal, seasonal sabzi and egg bhurji."
            dinner = "Moong dal khichdi with cucumber salad."
        elif meat:
            breakfast = "Vegetable poha with fruit."
            lunch = "Chapati with seasonal sabzi and well-cooked chicken."
            dinner = "Moong dal khichdi with cucumber salad."
        else:
            breakfast = "Vegetable poha with curd if tolerated."
            lunch = "Chapati with chana masala and seasonal sabzi."
            dinner = "Moong dal khichdi with cucumber salad and curd if tolerated."
        snack_options = ["Seasonal fruit", "Roasted chana", "Plain curd if tolerated" if not vegan else "Steamed chickpeas"]
    else:
        if vegan:
            breakfast = "Vegetable upma with sambar or oats prepared with an unsweetened plant drink."
            lunch = "Rice or chapati with dal, seasonal vegetables and chickpeas."
            dinner = "Vegetable khichdi with a side of beans or tofu."
        elif eggs:
            breakfast = "Vegetable upma with one well-cooked egg."
            lunch = "Rice or chapati with dal, seasonal vegetables and egg curry."
            dinner = "Vegetable khichdi with a side of dal."
        elif meat:
            breakfast = "Vegetable upma with seasonal fruit."
            lunch = "Rice or chapati with dal, vegetables and well-cooked chicken or fish."
            dinner = "Vegetable khichdi with a side of dal."
        else:
            breakfast = "Vegetable upma with fruit."
            lunch = "Rice or chapati with dal, seasonal vegetables and curd if tolerated."
            dinner = "Vegetable khichdi with a side of beans or paneer."
        snack_options = ["Seasonal fruit", "Roasted chana", "Plain curd if tolerated" if not vegan else "Steamed chickpeas"]

    menu = [
        ("Breakfast", breakfast),
        ("Lunch", lunch),
        ("Snack", snack_options[0]),
        ("Second snack", snack_options[1]),
        ("Dinner", dinner),
    ]
    if meals_per_day == "3 meals":
        chosen = [menu[0], menu[1], menu[4]]
    elif meals_per_day == "3 meals + 1 snack":
        chosen = [menu[0], menu[1], menu[2], menu[4]]
    else:
        chosen = menu

    budget_note = (
        "Choose seasonal vegetables, local fruit, rice, dal and roasted chana to keep costs lower."
        if "Affordable" in budget else
        "Rotate pulses, vegetables, grains and protein choices through the week."
    )
    output = "### Sample one-day meal plan\n" + "\n".join(
        f"**{name}:** {meal}" for name, meal in chosen
    )
    output += f"\n\n**Budget tip:** {budget_note}"
    if avoid_text:
        output += f"\n\n**Foods you asked to avoid:** {avoid_text}. Review ingredient lists and preparation methods before eating."
    if additional_request.strip():
        output += f"\n\n**Your extra preference:** {additional_request.strip()}. Adjust the listed options to this preference where possible."
    output += (
        "\n\nAdjust portions to your appetite and activity. This is general meal inspiration, "
        "not a calorie target, medical diet, or acne treatment."
    )
    return output

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
        '<div class="section-heading">'
        '<div class="eyebrow">BEAUTY TIPS</div>'
        '<h1 class="hero-title" style="font-size:34px">A routine that fits your skin.</h1>'
        '<p class="hero-copy">Choose your skin type and concerns. Get a simple morning and evening routine without unnecessary product recommendations.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
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
                "Your skincare question (optional)",
                placeholder="Example: I have oily skin. Do I still need moisturizer?",
                height=105,
                key="beauty_question",
            )
        generate_beauty = st.button(
            "Create my skincare routine",
            type="primary",
            use_container_width=True,
            key="generate_beauty_routine",
        )

    if generate_beauty:
        concern_text = ", ".join(concerns) if concerns else "General skin care"
        prompt = (
            "Create a practical, beginner-friendly skincare routine using only the details below.\n"
            f"Skin type: {skin_type}\nConcerns: {concern_text}\nSensitivity: {sensitivity}\n"
            f"User question: {extra_question or 'No extra question'}\n\n"
            "Return these exact sections with concise steps:\n"
            "1. Skin profile summary (one sentence)\n"
            "2. Morning routine (3 numbered steps: gentle cleanse, suitable moisturizer guidance, SPF 30+ sunscreen)\n"
            "3. Evening routine (3 numbered steps)\n"
            "4. What to avoid (3 useful points)\n"
            "5. When to consult a dermatologist.\n"
            "Adapt moisturizer texture to skin type. For sensitive skin prefer fragrance-free basics. "
            "Do not prescribe medicines, claim that a product cures acne, recommend lemon/toothpaste/scrubs, "
            "or invent product brands. Answer the user's extra question directly. Use simple language."
        )
        with st.spinner("Preparing your skincare routine..."):
            try:
                if _acneclear_setting("GEMINI_API_KEY"):
                    st.session_state["acneclear_beauty_response"] = ask_acneclear_ai(prompt)
                    st.session_state["acneclear_beauty_mode"] = "AI-generated"
                else:
                    st.session_state["acneclear_beauty_response"] = offline_beauty_starter(
                        skin_type, concerns, sensitivity, extra_question
                    )
                    st.session_state["acneclear_beauty_mode"] = "Offline starter"
            except Exception:
                st.session_state["acneclear_beauty_response"] = offline_beauty_starter(
                    skin_type, concerns, sensitivity, extra_question
                )
                st.session_state["acneclear_beauty_mode"] = "Offline starter"

    if st.session_state.get("acneclear_beauty_response"):
        mode = st.session_state.get("acneclear_beauty_mode", "Starter")
        st.markdown(f"### Your skincare routine · {mode}")
        with st.container(border=True):
            st.markdown(st.session_state["acneclear_beauty_response"])
    if not _acneclear_setting("GEMINI_API_KEY"):
        st.caption("Offline starter mode is active. Add GEMINI_API_KEY in Streamlit Secrets to enable personalized AI answers.")
    st.info(
        "General skincare education only. No online routine can guarantee clear skin. "
        "For painful, persistent or scarring acne, consult a dermatologist."
    )

elif page == "Diet Planner":
    st.markdown(
        '<div class="section-heading">'
        '<div class="eyebrow">DIET PLANNER</div>'
        '<h1 class="hero-title" style="font-size:34px">Everyday meals, planned simply.</h1>'
        '<p class="hero-copy">Choose your food preference, cuisine and budget. The planner gives a practical one-day menu with the number of meals you selected.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
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
            "Food allergies or foods to avoid",
            placeholder="Example: peanuts, milk, mushrooms (leave blank if none)",
            key="diet_allergies",
        )
        diet_question = st.text_area(
            "Anything else to consider? (optional)",
            placeholder="Example: I stay in a hostel and need quick, affordable meals.",
            height=90,
            key="diet_question",
        )
        generate_diet = st.button(
            "Create my meal plan",
            type="primary",
            use_container_width=True,
            key="generate_diet_plan",
        )

    if generate_diet:
        prompt = (
            "Create a realistic ONE-DAY Indian meal plan. Follow these user constraints exactly.\n"
            f"Food preference: {preference}\nCuisine: {meal_style}\nBudget: {budget}\n"
            f"Requested meals: {meals_per_day}\nFoods to avoid / allergies: {allergies or 'None specified'}\n"
            f"Additional context: {diet_question or 'None'}\n\n"
            "Output exactly the requested meal count. Include Breakfast, Lunch and Dinner. "
            "Add one or two clearly named snack slots only if the user selected them. "
            "For each meal provide one practical dish and a short description of its main ingredients. "
            "Keep choices aligned with the cuisine and budget. Vegetarian means no meat or fish; "
            "Includes eggs may include eggs but not meat or fish; Vegan means no meat, fish, eggs, milk, curd, paneer, ghee or other animal products; "
            "Includes meat or fish may include well-cooked chicken/fish. "
            "Never include foods listed in the allergy/avoid field. Do not use calorie targets or unsupported nutrient numbers. "
            "Do not call the plan an acne cure or medical diet. If an allergy is complex or serious, say that ingredient labels and cross-contact must be checked. "
            "End with 2 short budget-friendly preparation tips."
        )
        with st.spinner("Preparing your meal plan..."):
            try:
                if _acneclear_setting("GEMINI_API_KEY"):
                    st.session_state["acneclear_diet_response"] = ask_acneclear_ai(prompt)
                    st.session_state["acneclear_diet_mode"] = "AI-generated"
                else:
                    st.session_state["acneclear_diet_response"] = offline_diet_starter(
                        preference, meal_style, meals_per_day, allergies, budget, diet_question
                    )
                    st.session_state["acneclear_diet_mode"] = "Offline starter"
            except Exception:
                st.session_state["acneclear_diet_response"] = offline_diet_starter(
                    preference, meal_style, meals_per_day, allergies, budget, diet_question
                )
                st.session_state["acneclear_diet_mode"] = "Offline starter"

    if st.session_state.get("acneclear_diet_response"):
        mode = st.session_state.get("acneclear_diet_mode", "Starter")
        st.markdown(f"### Your one-day menu · {mode}")
        with st.container(border=True):
            st.markdown(st.session_state["acneclear_diet_response"])
    if not _acneclear_setting("GEMINI_API_KEY"):
        st.caption("Offline starter mode is active. For personalized AI planning, add GEMINI_API_KEY in Streamlit Secrets.")
    st.info(
        "This is general meal inspiration, not a medical diet. If you have a medical condition, serious allergy "
        "or medicine-related dietary restriction, consult a qualified professional."
    )


elif page == "AI Chatbot":
    st.markdown(
        '<div class="section-heading">'
        '<div class="eyebrow">ACNECLEAR INTELLIGENCE</div>'
        '<h1 class="hero-title" style="font-size:36px">Your assistant. One conversation away.</h1>'
        '<p class="hero-copy">Ask everyday questions, understand your skin-wellness options, or get clear explanations of how AcneClear AI was built.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    api_ready = bool(_acneclear_setting("GEMINI_API_KEY"))
    status_text = "AI service connected" if api_ready else "AI key required to start chatting"
    status_color = "#16805D" if api_ready else "#9A6715"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;margin:0 0 15px 2px">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{status_color};display:inline-block"></span>'
        f'<span style="font-size:12px;font-weight:700;color:{status_color}">{status_text}</span>'
        f'<span style="color:#8190A3;font-size:12px;margin-left:auto">AcneClear AI assistant</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    focus_options = ["General AI", "Skin & wellness", "Project mentor"]
    if "acneclear_chat_focus" not in st.session_state:
        st.session_state["acneclear_chat_focus"] = "General AI"

    focus = st.radio(
        "Choose an assistant focus",
        focus_options,
        horizontal=True,
        key="acneclear_chat_focus",
        label_visibility="collapsed",
    )

    title_col, action_col = st.columns([0.78, 0.22], gap="small")
    with title_col:
        st.markdown(
            '<div class="eyebrow" style="margin:8px 0 8px 0">CONVERSATION</div>',
            unsafe_allow_html=True,
        )
    with action_col:
        if st.button("Start a new chat", key="assistant_new_chat", use_container_width=True):
            st.session_state["acneclear_chat_history"] = []
            st.session_state.pop("acneclear_pending_prompt", None)
            st.rerun()

    with st.container(height=540, border=True):
        if not st.session_state["acneclear_chat_history"]:
            st.markdown(
                '<div style="padding:45px 16px;text-align:center">'
                '<div style="font-size:12px;letter-spacing:2px;font-weight:800;color:#496A91">WELCOME TO ACNECLEAR AI</div>'
                '<h3 style="color:#17365D;margin:10px 0 6px 0">What would you like to ask?</h3>'
                '<p style="font-size:13px;color:#52657A">Type your question below. You can ask about your project, skin wellness, coding, learning or everyday topics.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        for item in st.session_state["acneclear_chat_history"]:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

    prompt = st.chat_input(
        "Ask AcneClear AI anything...",
        disabled=not api_ready,
    )
    if prompt:
        previous_history = st.session_state["acneclear_chat_history"][-8:]
        st.session_state["acneclear_chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking through your question..."):
                try:
                    focus_instruction = (
                        f"Optional response focus: {focus}. Use this only as a style/context hint. "
                        "Always answer the actual question, including questions outside this focus.\n\n"
                    )
                    reply = ask_acneclear_ai(focus_instruction + prompt, previous_history)
                except Exception as exc:
                    reply = (
                        "I couldn't get a response from the AI service. "
                        "Check the GEMINI_API_KEY in Streamlit Secrets and the available free-tier quota. "
                        f"Details: {type(exc).__name__}: {exc}"
                    )
                st.markdown(reply)
        st.session_state["acneclear_chat_history"].append({"role": "assistant", "content": reply})

st.markdown("---")
st.markdown('<div class="small-note">ACNECLEAR AI · Educational photo-retouching and wellness demo. Photo restoration is not a dedicated acne-removal model. Skincare and diet information is general education, not medical advice.</div>', unsafe_allow_html=True)