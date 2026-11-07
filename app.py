import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="CLIP Image Compare", page_icon="🖼️")

st.title("🖼️ Сравнение фотографий")
st.write("Загрузи две фотографии, чтобы сравнить их сходство.")

@st.cache_resource
def load_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return processor, model

file1 = st.file_uploader("Первая фотография", type=["jpg", "jpeg", "png", "jfif"])
file2 = st.file_uploader("Вторая фотография", type=["jpg", "jpeg", "png", "jfif"])

if file1 and file2:
    image1 = Image.open(file1).convert("RGB")
    image2 = Image.open(file2).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image1, caption="Первая фотография", use_container_width=True)

    with col2:
        st.image(image2, caption="Вторая фотография", use_container_width=True)

    if st.button("Сравнить фотографии"):
        with st.spinner("Сравниваем..."):
            processor, model = load_model()

            inputs = processor(images=[image1, image2], return_tensors="pt")

            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            similarity = torch.nn.functional.cosine_similarity(
                features[0].unsqueeze(0),
                features[1].unsqueeze(0)
            ).item()

        st.metric("Сходство изображений", f"{similarity * 100:.2f}%")
        st.caption("Это оценка сходства изображений CLIP, а не вероятность совпадения объектов.")
