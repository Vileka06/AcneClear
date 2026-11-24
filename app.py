from io import BytesIO
from pathlib import Path
import sys
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# Model files are downloaded once and kept outside Git.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
WEIGHT_PATH = MODEL_DIR / "GFPGANv1.3.pth"
WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


st.set_page_config(
    page_title="AcneClear AI",
    page_icon="✨",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background: #ffffff; color: #243B63; }
    .block-container { max-width: 850px; padding-top: 2.2rem; }
    h1, h2, h3 { color: #243B63 !important; }
    .intro { color: #64748b; margin-top: -8px; margin-bottom: 24px; }
    div[data-testid="stFileUploader"] {
        background: #f7f9fc;
        border: 1px dashed #aebed2;
        border-radius: 10px;
        padding: 12px;
    }
    div.stButton > button, div.stDownloadButton > button {
        background: #243B63;
        color: white;
        border: 0;
        border-radius: 7px;
        min-height: 42px;
        font-weight: 600;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #34527d;
        color: white;
    }
    hr { border-color: #e6ebf2; }
</style>
""", unsafe_allow_html=True)

st.title("AcneClear AI")
st.markdown(
    '<p class="intro">Upload a photo, enhance it, compare the result, and download it.</p>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_gfpgan():
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


def enhance_photo(photo: Image.Image) -> Image.Image:
    rgb = np.asarray(photo.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    restorer = load_gfpgan()
    _, _, restored = restorer.enhance(
        bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.6,
    )

    if restored is None:
        raise RuntimeError("No face could be restored. Try a clear, front-facing photo.")

    result_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


uploaded = st.file_uploader(
    "1. Upload your photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPG, PNG or WEBP. Maximum size: 10 MB.",
)

if uploaded is not None:
    photo_bytes = uploaded.getvalue()

    if len(photo_bytes) > MAX_UPLOAD_BYTES:
        st.error("Please upload an image smaller than 10 MB.")
        st.stop()

    try:
        original = ImageOps.exif_transpose(
            Image.open(BytesIO(photo_bytes))
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This image could not be opened. Please choose another photo.")
        st.stop()

    st.image(original, caption="Original photo", use_container_width=True)

    if st.button("2. Enhance photo", type="primary", use_container_width=True):
        try:
            with st.spinner("Enhancing your photo... The first run may take longer."):
                st.session_state["acneclear_result"] = enhance_photo(original)
                st.session_state["acneclear_source"] = uploaded.name
        except Exception as error:
            st.session_state.pop("acneclear_result", None)
            st.error("Could not enhance this photo. Try a clear, front-facing image.")
            st.caption(f"Technical detail: {type(error).__name__}: {error}")

    result = st.session_state.get("acneclear_result")
    if result is not None and st.session_state.get("acneclear_source") == uploaded.name:
        st.divider()
        st.subheader("3. Before and after")

        before, after = st.columns(2)
        with before:
            st.image(original, caption="Before", use_container_width=True)
        with after:
            st.image(result, caption="After", use_container_width=True)

        download = BytesIO()
        result.save(download, format="PNG")
        st.download_button(
            "4. Download enhanced photo",
            data=download.getvalue(),
            file_name="acneclear_enhanced.png",
            mime="image/png",
            use_container_width=True,
        )
else:
    st.session_state.pop("acneclear_result", None)
    st.session_state.pop("acneclear_source", None)

st.divider()
st.caption(
    "Powered by GFPGAN. Photos are processed in memory and are not saved by this app. "
    "GFPGAN is a general face-restoration model, not an acne-specific treatment or medical diagnosis."
)
