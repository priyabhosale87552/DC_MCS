import streamlit as st
import requests

st.title("📄 Document Classification using Modified Cosine Similarity")

# Text input
text_input = st.text_area("✍️ Enter text to classify")

# File upload
uploaded_file = st.file_uploader("📂 Or upload a text file", type=["txt"])

if st.button("Classify"):
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")

    if text_input.strip():
        try:
            response = requests.post("http://127.0.0.1:5000/classify", json={"text": text_input})
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Predicted Category: **{result['prediction']}**")
                st.info(result["explanation"])
            else:
                st.error("❌ Error from backend")
        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")
    else:
        st.warning("⚠️ Please enter some text or upload a file.")
