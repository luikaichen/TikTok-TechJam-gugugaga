from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.predict import DEFAULT_CHECKPOINT, PREPROCESS, load_model

DEMO_THRESHOLD = 0.05


@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(DEFAULT_CHECKPOINT, device)
    return model, device


def predict_p_ai(image: Image.Image, model, device) -> float:
    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    return float(probs[0, 0].item())


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Robust AI-Generated Image Detector")
st.caption("TikTok TechJam 2026 · Track 5 · Generalisation V2")

st.markdown(
    '''
Upload an image to estimate **P(AI)** using the frozen Generalisation V2 model.

- Raw model output: **P(AI) = softmax(logits)[0]**
- Class `0` = AIGC
- Class `1` = REAL
- Demo calibrated threshold: **0.05**
'''
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption=uploaded.name, use_column_width=True)

    with st.spinner("Running Generalisation V2..."):
        model, device = get_model()
        p_ai = predict_p_ai(image, model, device)

    demo_label = "AIGC" if p_ai >= DEMO_THRESHOLD else "REAL"

    left, right = st.columns(2)
    with left:
        st.metric("AI Probability", f"{p_ai:.2%}")
    with right:
        st.metric("Demo Decision", demo_label)

    st.progress(min(max(p_ai, 0.0), 1.0))

    if demo_label == "AIGC":
        st.warning(
            f"Predicted AIGC at the calibrated demo threshold "
            f"({DEMO_THRESHOLD:.2f})."
        )
    else:
        st.success(
            f"Predicted REAL at the calibrated demo threshold "
            f"({DEMO_THRESHOLD:.2f})."
        )

    with st.expander("Model details"):
        st.write(f"Device: `{device}`")
        st.write(f"Checkpoint: `{DEFAULT_CHECKPOINT.name}`")
        st.write(f"Raw P(AI): `{p_ai:.8f}`")
        st.write(f"Demo threshold: `{DEMO_THRESHOLD}`")
        st.write(
            "The calibrated threshold is demonstration decision logic only; "
            "the raw probability remains unchanged."
        )

st.divider()

st.markdown(
    '''
### Key finding

> **Transformation robustness and domain robustness are different problems.**

Generalisation V2 was selected to preserve robustness to image transformations
while improving cross-domain REAL-image generalisation.
'''
)
