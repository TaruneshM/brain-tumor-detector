import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import io
import os

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="assets/logo.png",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stMetric { background: #1a1d25; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

CLASS_COLORS = {
    'tumor':        (255, 68,  68),
    'tumor_type_1': (255, 140, 0),
    'tumor_type_2': (155, 89,  182),
}

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    return YOLO(model_path)

model = load_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.title("Brain Tumor Detector")
st.caption("YOLOv8 · Trained on Google Colab · 3 classes: tumor · tumor_type_1 · tumor_type_2")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
                            format="%.2f")
    iou_thresh  = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05,
                            format="%.2f")
    show_labels = st.toggle("Show Labels on Image", value=True)
    show_conf   = st.toggle("Show Confidence on Image", value=True)

    st.divider()
    st.markdown("**Legend**")
    for cls, rgb in CLASS_COLORS.items():
        hex_col = "#{:02x}{:02x}{:02x}".format(*rgb)
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{hex_col};margin-right:8px"></span>'
            f'`{cls}`',
            unsafe_allow_html=True
        )

# ── Upload ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Image")
    uploaded = st.file_uploader("Upload a brain MRI scan",
                                type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original scan", use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
with col2:
    st.subheader("Detection Result")

    if not uploaded:
        st.info(" Upload a brain scan on the left to get started.")
    else:
        with st.spinner("Running YOLOv8 inference..."):
            results = model(img, conf=conf_thresh, iou=iou_thresh)
            result  = results[0]

        # Draw boxes manually for color control
        out_img = img.copy()
        draw    = ImageDraw.Draw(out_img)

        detections = []
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            color = CLASS_COLORS.get(cls_name, (79, 142, 247))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            parts = []
            if show_labels: parts.append(cls_name)
            if show_conf:   parts.append(f"{conf:.0%}")
            label = " ".join(parts)

            if label:
                tw = len(label) * 7 + 6
                draw.rectangle([x1, y1 - 22, x1 + tw, y1], fill=color)
                draw.text((x1 + 3, y1 - 20), label, fill="white")

            detections.append(dict(Class=cls_name,
                                   Confidence=f"{conf:.1%}",
                                   Box=f"({x1},{y1})→({x2},{y2})",
                                   W=x2-x1, H=y2-y1))

        st.image(out_img, caption=f"{len(detections)} detection(s)", use_container_width=True)

# ── Metrics + Table ────────────────────────────────────────────────────────────
if uploaded and 'detections' in dir():
    st.divider()

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detections", len(detections))
    counts = {}
    for d in detections:
        counts[d['Class']] = counts.get(d['Class'], 0) + 1
    m2.metric("tumor",        counts.get('tumor', 0))
    m3.metric("tumor_type_1", counts.get('tumor_type_1', 0))
    m4.metric("tumor_type_2", counts.get('tumor_type_2', 0))

    # Table
    if detections:
        st.subheader("Detection Details")
        st.dataframe(detections, width='stretch', hide_index=True)

        # Download annotated image
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="tumor_detection_result.png",
            mime="image/png"
        )
    else:
        st.warning("No tumors detected. Try lowering the confidence threshold in the sidebar.")