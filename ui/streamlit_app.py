from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("LEGAL_API_URL", "http://localhost:8000/v1/query")

st.set_page_config(page_title="Legal Intelligence Engine", layout="wide")
st.title("Legal Intelligence Engine")
st.caption("Retrieval + evidence + answer generation in one interface")

query = st.text_area("Enter a legal question", height=150, placeholder="مثال: ما هي شروط القبض في حالة التلبس؟")

if st.button("Run query") and query.strip():
    try:
        payload = {"query": query, "top_k": 5}
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code != 200:
            st.error(f"API error: {response.status_code} - {response.text}")
        else:
            data = response.json()
            st.subheader("Answer")
            st.write(data.get("answer", "No answer returned."))
            with st.expander("Sources"):
                for item in data.get("sources", []):
                    st.markdown(f"- **{item.get('law_name', 'Unknown')}** / Article {item.get('article_id', '')}: {item.get('text', '')[:400]}")
    except requests.RequestException as exc:
        st.warning(f"API is not running. Start the backend first or set LEGAL_API_URL. Error: {exc}")
