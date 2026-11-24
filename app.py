from io import BytesIO
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
WEIGHT_PATH = MODEL_DIR / "GFPGANv1.3.pth"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_GFPGAN_RESTORER = None

st.set_page_config(page_title="AcneClear AI | Photo Enhancement", page_icon="✳", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --navy:#243B63; --muted:#65758B; --line:#E3EAF2; --pale:#EAF1F8; }
.stApp { background:#F8FAFC; color:#243B63; font-family:'DM Sans',sans-serif; }
.block-container { max-width:1120px; padding-top:2rem; padding-bottom:4rem; }
#MainMenu, footer {visibility:hidden;}
h1,h2,h3 {font-family:'Manrope',sans-serif !important;color:var(--navy) !important;}
.brand {background:#fff;border:1px solid var(--line);border-radius:15px;padding:15px 18px;margin-bottom:28px;display:flex;align-items:center;gap:12px;}
.brand-mark {width:42px;height:42px;border-radius:12px;background:var(--navy);color:#fff;display:grid;place-items:center;font-size:22px;}
.brand-name {font:800 19px Manrope;color:var(--navy);}
.brand-sub {font-size:10px;letter-spacing:1.5px;color:#8190A3;margin-top:3px;}
.hero {text-align:center;max-width:720px;margin:0 auto 30px;}
.eyebrow {font-size:10px;font-weight:800;letter-spacing:2px;color:#7890AB;}
.hero p {font-size:14px;line-height:1.7;color:var(--muted);}
.panel {background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px;}
div[data-testid="stFileUploader"] {background:#F8FAFC;border:1px dashed #B8C8DA;border-radius:12px;padding:12px;}
div.stButton>button,div.stDownloadButton>button {background:var(--navy);color:#fff;border:1px solid var(--navy);border-radius:9px;min-height:44px;font-weight:700;}
div.stButton>button:hover,div.stDownloadButton>button:hover {background:#34527D;color:#fff;border-color:#34527D;}
div.stButton>button p,div.stDownloadButton>button p {color:#fff;}
div[data-testid="stImage"] img {border-radius:12px;}
.note {font-size:11px;color:#8290A1;line-height:1.6;text-align:center;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
  <div class="brand-mark">✳</div>
  <div><div class="brand-name">AcneClear AI</div><div class="brand-sub">PHOTO ENHANCEMENT</div></div>
</div>
<div class="hero">
  <div class="eyebrow">SIMPLE. PRIVATE. PHOTO-FOCUSED.</div>
  <h1>Enhance your photo.<br>Keep your natural details.</h1>
  <p>Upload a clear face photo to restore facial details and apply gentle texture smoothing. Preview the result and download your enhanced image.</p>
</div>
""", unsafe_allow_html=True)


def load_restorer():
    global _GFPGAN_RESTORER
    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from gfpgan import GFPGANer
    if not WEIGHT_PATH.exists():
        with st.spinner("Downloading the GFPGAN v1.3 model for first-time use..."):
            urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
    _GFPGAN_RESTORER = GFPGANer(
        model_path=str(WEIGHT_PATH), upscale=2, arch="clean",
        channel_multiplier=2, bg_upsampler=None,
    )
    return _GFPGAN_RESTORER


def soften_texture(image_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=50, sigmaSpace=50)
    blended = cv2.addWeighted(bgr, 0.78, smooth, 0.22, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def enhance_photo(original: Image.Image) -> Image.Image:
    rgb = np.asarray(original.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restorer = load_restorer()
    _, _, restored_bgr = restorer.enhance(
        bgr, has_aligned=False, only_center_face=False, paste_back=True, weight=0.6
    )
    if restored_bgr is None:
        raise RuntimeError("No clear face could be restored. Try a front-facing, well-lit photo.")
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(soften_texture(restored_rgb))


if "acneclear_photo_id" not in st.session_state:
    st.session_state.acneclear_photo_id = None
if "acneclear_result" not in st.session_state:
    st.session_state.acneclear_result = None

with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png", "webp"], help="JPG, PNG or WEBP. Maximum file size: 10 MB.")
    if uploaded is not None:
        photo_bytes = uploaded.getvalue()
        photo_id = (uploaded.name, len(photo_bytes))
        if len(photo_bytes) > MAX_UPLOAD_BYTES:
            st.error("Please choose an image smaller than 10 MB.")
            st.stop()
        if st.session_state.acneclear_photo_id != photo_id:
            st.session_state.acneclear_photo_id = photo_id
            st.session_state.acneclear_result = None
        try:
            original = ImageOps.exif_transpose(Image.open(BytesIO(photo_bytes))).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be opened as an image. Please choose another photo.")
            st.stop()
        st.image(original, caption="Original photo", use_container_width=True)
        if st.button("Enhance photo", type="primary", use_container_width=True):
            try:
                with st.spinner("Enhancing your photo. The first run may take longer while the model loads..."):
                    st.session_state.acneclear_result = enhance_photo(original)
                st.success("Your enhanced photo is ready.")
            except Exception as exc:
                st.session_state.acneclear_result = None
                st.error("The photo could not be enhanced. Try a clear, front-facing photo.")
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
        if st.session_state.acneclear_result is not None:
            st.markdown("---")
            st.subheader("Your enhanced photo")
            original_col, result_col = st.columns(2, gap="large")
            with original_col:
                st.image(original, caption="Original", use_container_width=True)
            with result_col:
                st.image(st.session_state.acneclear_result, caption="Enhanced", use_container_width=True)
            output = BytesIO()
            st.session_state.acneclear_result.save(output, format="PNG")
            st.download_button("Download enhanced photo", data=output.getvalue(), file_name="acneclear_enhanced.png", mime="image/png", use_container_width=True)
    else:
        st.session_state.acneclear_photo_id = None
        st.session_state.acneclear_result = None
        st.info("Choose a photo to get started.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="note">Your photo is processed in memory and is not saved by this app. '
    'GFPGAN is a general face-restoration model, not an acne-specific removal or medical diagnosis system. '
    'Results vary by image quality and face visibility.</div>',
    unsafe_allow_html=True,
)
