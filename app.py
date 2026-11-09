import io
import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Visual Search", page_icon="🔎", layout="wide")
st.title("🔎 Визуальный поиск товаров")
st.caption("Поиск по изображениям в каталоге, который был явно добавлен в API.")

api_url = st.sidebar.text_input(
    "Адрес API",
    value=os.getenv("VISUAL_SEARCH_API_URL", "http://127.0.0.1:8000"),
).rstrip("/")
api_key = st.sidebar.text_input(
    "API key (если включён на сервере)",
    value=os.getenv("VISUAL_SEARCH_API_KEY", ""),
    type="password",
)
headers = {"X-API-Key": api_key} if api_key else {}

tab_search, tab_compare, tab_status = st.tabs(["Поиск по каталогу", "Сравнение", "Состояние API"])

with tab_search:
    reference = st.file_uploader("Фотография товара", type=["jpg", "jpeg", "png", "webp"], key="search")
    top_k = st.slider("Количество результатов", min_value=5, max_value=50, value=20)
    clip_weight = st.slider("Вес CLIP", 0.0, 1.0, 0.5, 0.05)
    if st.button("Найти похожие товары", type="primary", disabled=reference is None):
        if clip_weight == 0:
            dino_weight = 1.0
        elif clip_weight == 1:
            dino_weight = 0.0
        else:
            dino_weight = 1.0 - clip_weight
        with st.spinner("Сравниваем изображение с каталогом..."):
            try:
                response = requests.post(
                    f"{api_url}/search",
                    headers=headers,
                    files={"reference": (reference.name, reference.getvalue(), reference.type or "image/jpeg")},
                    data={"top_k": top_k, "clip_weight": clip_weight, "dino_weight": dino_weight},
                    timeout=180,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                st.error(f"Не удалось выполнить поиск: {exc}")
                payload = None
        if payload is not None:
            results = payload.get("results", [])
            st.write(f"Найдено результатов: {payload.get('count', len(results))}")
            if not results:
                st.info(payload.get("message", "В каталоге пока нет изображений."))
            for item in results:
                col_image, col_text = st.columns([1, 3])
                with col_image:
                    if item.get("image_url"):
                        st.image(item["image_url"], use_container_width=True)
                with col_text:
                    st.subheader(item.get("title") or "Без названия")
                    st.write(item.get("price") or "Цена не указана")
                    st.caption(f"Сходство: {item.get('similarity_percent', 0):.2f}%")
                    if item.get("listing_url"):
                        st.markdown(f"[Открыть объявление]({item['listing_url']})")
                    if item.get("image_url"):
                        st.caption(item["image_url"])

with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        first = st.file_uploader("Эталон", type=["jpg", "jpeg", "png", "webp"], key="first")
    with col2:
        second = st.file_uploader("Кандидат", type=["jpg", "jpeg", "png", "webp"], key="second")
    if first and second:
        left, right = st.columns(2)
        left.image(first, caption="Эталон", use_container_width=True)
        right.image(second, caption="Кандидат", use_container_width=True)
        if st.button("Сравнить", key="compare"):
            try:
                response = requests.post(
                    f"{api_url}/rank",
                    headers=headers,
                    files={
                        "reference": (first.name, first.getvalue(), first.type or "image/jpeg"),
                        "candidate1": (second.name, second.getvalue(), second.type or "image/jpeg"),
                    },
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()["results"][0]
                st.metric("Общее сходство", f"{result['similarity_percent']:.2f}%")
                st.write(f"CLIP: {result['clip_similarity']:.4f} · DINOv2: {result['dino_similarity']:.4f}")
                st.caption("Оценка сходства — сигнал ранжирования, а не вероятность идентичности.")
            except requests.RequestException as exc:
                st.error(f"Не удалось сравнить изображения: {exc}")

with tab_status:
    if st.button("Проверить API"):
        try:
            response = requests.get(f"{api_url}/health", headers=headers, timeout=15)
            response.raise_for_status()
            st.json(response.json())
            stats = requests.get(f"{api_url}/stats", headers=headers, timeout=15)
            stats.raise_for_status()
            st.json(stats.json())
        except requests.RequestException as exc:
            st.error(f"API недоступен: {exc}")
