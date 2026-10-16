import os
import torch
import pandas as pd
from PIL import Image
import streamlit as st

from src.model import CNN
from src.dataset import CLASSES, get_inference_transforms

# 1. Page Configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Device Configuration & Model Loading (Cached)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model(weights_path: str = "weights/cifar10_cnn.pth"):
    device = get_device()
    model = CNN().to(device)

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        st.warning(f"Weights file not found at `{weights_path}`. Running with initialized weights.")

    model.eval()
    return model, device


model, device = load_model()
transform = get_inference_transforms()

# Class Icons for Visual Polish
CLASS_ICONS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐶",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚚"
}

# 3. Sidebar Information
with st.sidebar:
    st.title("ℹ️ About the Project")
    st.markdown(
        """
        A custom **3-Stage Convolutional Neural Network (CNN)** trained from scratch using **PyTorch** on the CIFAR-10 dataset.
        
        ---
        ### 🧠 Architecture Specs
        * **Conv Blocks**: 3 stages (32 → 64 → 128 channels)
        * **Regularization**: `BatchNorm2d` after each conv, `Dropout2d(0.25)` on feature maps, `Dropout(0.5)` on dense layers
        * **Augmentation**: Random Crop + Random Horizontal Flip
        * **Parameters**: 620,810 trainable weights (~2.37 MB)
        * **Hardware**: Running on **`{}`**
        """.format(str(device).upper())
    )
    st.markdown("---")
    st.markdown("🔗 [View on GitHub](https://github.com)")

# 4. Main Application Interface
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown(
    "Upload any picture to classify it into one of 10 categories: "
    "**Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, or Truck**."
)

input_method = st.radio("Choose input method:", ("Upload File", "Use Camera"), horizontal=True)

image = None
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Take a photo...")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

if image is not None:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Predictions")

        # Preprocessing & Inference
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Top Prediction
        top_prob, top_idx = torch.topk(probabilities, 1)
        top_class = CLASSES[top_idx.item()]
        icon = CLASS_ICONS.get(top_class, "🏷️")

        st.success(f"**Top Result:** {icon} **{top_class.capitalize()}** ({top_prob.item() * 100:.1f}%)")

        st.markdown("#### Top 3 Confidences:")
        top3_probs, top3_indices = torch.topk(probabilities, 3)

        for prob, idx in zip(top3_probs, top3_indices):
            cname = CLASSES[idx.item()]
            pct = prob.item()
            st.write(f"{CLASS_ICONS.get(cname, '•')} **{cname.capitalize()}** — `{pct * 100:.2f}%`")
            st.progress(pct)

    # Complete Probabilities Expander
    with st.expander("📊 View Complete 10-Class Probability Breakdown"):
        prob_dict = {
            "Class": [f"{CLASS_ICONS.get(c, '')} {c.capitalize()}" for c in CLASSES],
            "Probability (%)": [round(float(p) * 100, 2) for p in probabilities]
        }
        df_probs = pd.DataFrame(prob_dict).sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_probs, use_container_width=True)
else:
    st.info("👆 Upload an image or snap a photo above to test the model.")
