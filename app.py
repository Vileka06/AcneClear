from io import BytesIO
from pathlib import Path
import urllib.request

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
GFPGAN_PATH = MODEL_DIR / "GFPGANv1.3.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
MAX_UPLOAD_MB = 10
LATENT_SIZE = 100
IMAGE_SIZE = 64

st.set_page_config(page_title="AcneClear AI", page_icon="AC", layout="wide")

st.markdown("""
<style>
.stApp {background:#F7F9FC;color:#172B4D}
.block-container {max-width:1000px;padding-top:2rem}
h1,h2,h3 {color:#243B63}
div.stButton>button,div.stDownloadButton>button {
 background:#243B63;color:white;border:0;border-radius:8px;
 min-height:42px;font-weight:700
}
div.stButton>button:hover,div.stDownloadButton>button:hover {
 background:#34527D;color:white
}
div[data-testid="stFileUploader"] {
 background:white;border:1px dashed #AFC1D5;border-radius:12px;padding:12px
}
.panel {background:white;border:1px solid #DCE5EF;border-radius:14px;padding:18px}
.note {color:#68788D;font-size:13px}
</style>
""", unsafe_allow_html=True)

st.title("AcneClear AI")
st.write("A simple GAN-based photo generation and photo enhancement project.")
st.caption("Mini Project · Generative Adversarial Networks (GANs)")

generate_tab, enhance_tab = st.tabs(["Generate Photo", "Enhance Photo"])

class DCGANGenerator(nn.Module):
    """Standard 64x64 DCGAN generator. Checkpoint must use this architecture."""
    def __init__(self, latent_dim=LATENT_SIZE, channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8), nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.net(noise)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("generator_state_dict", "state_dict", "generator", "G"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint must contain a generator state dictionary.")


def make_generated_image(checkpoint_bytes):
    try:
        checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu")
    state = extract_state_dict(checkpoint)
    generator = DCGANGenerator()
    generator.load_state_dict(state, strict=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, LATENT_SIZE, 1, 1)
        output = generator(noise)[0].cpu().clamp(-1, 1)
        output = ((output + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(output)


@st.cache_resource(show_spinner=False)
def load_face_restorer():
    # Compatibility shim for BasicSR versions that import this older torchvision path.
    import sys
    import torchvision.transforms.functional as tv_functional
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from gfpgan import GFPGANer

    if not GFPGAN_PATH.exists():
        urllib.request.urlretrieve(GFPGAN_URL, GFPGAN_PATH)

    return GFPGANer(
        model_path=str(GFPGAN_PATH),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def enhance_photo(photo):
    import cv2
    rgb = np.asarray(photo.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restorer = load_face_restorer()
    _, _, restored = restorer.enhance(
        bgr, has_aligned=False, only_center_face=False,
        paste_back=True, weight=0.6
    )
    if restored is None:
        raise RuntimeError("No face was detected. Please upload a clear, front-facing portrait.")
    return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))


with generate_tab:
    st.subheader("Generate a new photo")
    st.write("Upload your trained DCGAN generator checkpoint to create a new image.")
    st.info(
        "The generator checkpoint is not stored in this GitHub repository. "
        "Upload your trained .pth file below. It must match the standard 64×64 DCGAN "
        "generator architecture used by this app."
    )
    generator_file = st.file_uploader(
        "DCGAN generator checkpoint (.pth)", type=["pth", "pt"], key="generator_checkpoint"
    )
    if st.button("Generate Photo", disabled=generator_file is None, key="generate_photo"):
        try:
            with st.spinner("Generating image with the GAN..."):
                result = make_generated_image(generator_file.getvalue())
            st.image(result, caption="Generated image", use_container_width=True)
            output = BytesIO()
            result.save(output, format="PNG")
            st.download_button(
                "Download generated photo", output.getvalue(),
                file_name="acneclear_generated.png", mime="image/png"
            )
        except Exception as exc:
            st.error("Could not load this generator checkpoint. Check that it matches the DCGAN architecture.")
            st.caption(str(exc))

with enhance_tab:
    st.subheader("Enhance an existing photo")
    st.write("Upload a clear portrait. GFPGAN will restore facial details using a pretrained GAN model.")
    uploaded_photo = st.file_uploader(
        "Choose a portrait image", type=["jpg", "jpeg", "png", "webp"], key="enhance_upload"
    )
    if uploaded_photo is not None:
        if uploaded_photo.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Please choose an image smaller than {MAX_UPLOAD_MB} MB.")
        else:
            try:
                original = ImageOps.exif_transpose(Image.open(uploaded_photo)).convert("RGB")
                st.image(original, caption="Original photo", use_container_width=True)
                if st.button("Enhance Photo", key="enhance_photo"):
                    with st.spinner("Restoring your photo with GFPGAN..."):
                        enhanced = enhance_photo(original)
                    left, right = st.columns(2)
                    with left:
                        st.image(original, caption="Before", use_container_width=True)
                    with right:
                        st.image(enhanced, caption="After", use_container_width=True)
                    output = BytesIO()
                    enhanced.save(output, format="PNG")
                    st.download_button(
                        "Download enhanced photo", output.getvalue(),
                        file_name="acneclear_enhanced.png", mime="image/png"
                    )
            except Exception as exc:
                st.error("The image could not be processed.")
                st.caption(str(exc))

st.divider()
st.caption(
    "GFPGAN is a general face-restoration model, not a dedicated acne-removal model. "
    "Enhancement results vary and should not be treated as medical advice."
)
