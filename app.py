import base64
import io
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

@st.cache_resource(show_spinner=False)
def load_restorer():
    from gfpgan import GFPGANer
    if not WEIGHT_PATH.exists():
        with st.spinner("Downloading the official GFPGAN v1.3 model (first run only)…"):
            urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
    return GFPGANer(
        model_path=str(WEIGHT_PATH),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

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

# Sidebar navigation
with st.sidebar:
    st.markdown("## AcneClear AI")
    page = st.radio(
        "Choose a section",
        ["Acne Remover", "Beauty Tips", "Diet Planner"],
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
    st.markdown('<div class="eyebrow">THE DAILY SKIN EDIT</div><h1 class="hero-title">Small habits.<br>Gentler skin care.</h1><p class="hero-copy">A practical routine for acne-prone skin, with simple steps you can build into your day. Keep it gentle and introduce new products one at a time.</p>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1608248543803-ba4f8c70ae0b?auto=format&fit=crop&w=1600&q=85", caption="A simple, gentle skincare routine", use_container_width=True)
    st.markdown('<div class="pill">MORNING TO EVENING</div>', unsafe_allow_html=True)
    morning, evening = st.columns(2, gap="large")
    with morning:
        st.markdown('<div class="panel"><div class="panel-title">Morning routine</div><div class="panel-sub">A simple routine to protect your skin during the day.</div></div>', unsafe_allow_html=True)
        for item in ["Use a gentle cleanser or rinse with lukewarm water.", "Apply a non-comedogenic moisturizer.", "Use broad-spectrum SPF 30+ sunscreen.", "Avoid scrubbing or picking blemishes."]:
            st.markdown(f"- {item}")
    with evening:
        st.markdown('<div class="panel"><div class="panel-title">Evening routine</div><div class="panel-sub">Remove the day gently and support your skin barrier.</div></div>', unsafe_allow_html=True)
        for item in ["Remove makeup and sunscreen gently.", "Cleanse without harsh scrubbing.", "Moisturize with a product that suits your skin.", "Wash pillowcases regularly and clean phone screens."]:
            st.markdown(f"- {item}")
    st.info("Persistent, painful, scarring, or distressing acne is worth discussing with a dermatologist. These tips are general education, not a diagnosis.")

elif page == "Diet Planner":
    st.markdown('<div class="eyebrow">THE EVERYDAY FOOD GUIDE</div><h1 class="hero-title">Eat well.<br>Keep it balanced.</h1><p class="hero-copy">Explore practical meal ideas built around variety, nourishment, and foods you enjoy. No single meal or diet guarantees clear skin.</p>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=1600&q=85", caption="A colourful, balanced plate", use_container_width=True)
    preference = st.selectbox("Choose a meal preference", ["Vegetarian", "Includes eggs", "Includes meat or fish"])
    if preference == "Vegetarian":
        meals = [
            ("Breakfast", "Oats with nuts and fruit, or idli with sambar."),
            ("Lunch", "Rice or millet, dal, vegetables, and curd if tolerated."),
            ("Snack", "A fruit with a small handful of nuts."),
            ("Dinner", "Chapati with beans, paneer or tofu, and vegetables."),
        ]
    elif preference == "Includes eggs":
        meals = [
            ("Breakfast", "Eggs with whole-grain toast and fruit, or oats with nuts."),
            ("Lunch", "Rice or millet, dal, vegetables, and boiled egg."),
            ("Snack", "Fruit and unsweetened yogurt if tolerated."),
            ("Dinner", "Chapati with egg curry or beans and vegetables."),
        ]
    else:
        meals = [
            ("Breakfast", "Oats with nuts and fruit, or eggs with whole-grain toast."),
            ("Lunch", "Rice or millet, vegetables, dal, and fish or chicken."),
            ("Snack", "Fruit and a small handful of nuts."),
            ("Dinner", "Chapati with fish, chicken, beans, or lentils and vegetables."),
        ]
    for meal, suggestion in meals:
        st.markdown(f'<div class="panel" style="margin-bottom:10px"><div class="eyebrow">{meal}</div><div class="panel-title">{suggestion}</div></div>', unsafe_allow_html=True)
    st.markdown("**Helpful habits**")
    st.write("- Drink water regularly and eat a variety of foods.")
    st.write("- Notice whether a particular food seems to coincide with flare-ups; avoid overly restrictive diets.")
    st.write("- Some people report acne triggers from high-glycemic foods or certain dairy products, but responses differ.")
    st.caption("This planner is general information, not a treatment plan. Follow any medical or dietary advice you have already received.")

st.markdown("---")
st.markdown('<div class="small-note">ACNECLEAR AI · Educational photo-retouching and wellness demo. Photo restoration is not a dedicated acne-removal model. Skincare and diet information is general education, not medical advice.</div>', unsafe_allow_html=True)