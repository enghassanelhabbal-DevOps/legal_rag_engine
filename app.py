from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_legal_documents() -> list[dict[str, Any]]:
    candidates = [
        ROOT / "legal_documents.json",
        ROOT / "data" / "legal_documents.json",
        ROOT / "data" / "normalized" / "documents.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload
    return [
        {
            "document_id": "sample-1",
            "law_name": "قانون الإجراءات الجنائية",
            "article_id": "1",
            "content": "تُعد الإجراءات الجنائية من أهم الضمانات في القضايا الجنائية، وتخضع للشرط القانوني في حالة التلبس.",
            "metadata": {"title": "المادة 1 - قانون الإجراءات الجنائية"},
        },
        {
            "document_id": "sample-2",
            "law_name": "قانون العقوبات",
            "article_id": "2",
            "content": "يلتزم القاضي في حالة التلبس بالتحقق من الواقعة قبل إصدار القرار، مع مراعاة حقوق المتهم.",
            "metadata": {"title": "المادة 2 - قانون العقوبات"},
        },
    ]


@st.cache_data
def build_bm25_index() -> tuple[list[str], list[dict[str, Any]]]:
    docs = load_legal_documents()
    corpus: list[str] = []
    records: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc.get("metadata") or {}
        title = str(metadata.get("title", ""))
        text = str(doc.get("content", ""))
        combined = f"{title} {text}".strip()
        corpus.append(combined)
        records.append(
            {
                "document_id": str(doc.get("document_id") or doc.get("id") or len(records)),
                "law_name": str(doc.get("law_name") or metadata.get("law_name") or "غير محدد"),
                "article_id": str(doc.get("article_id") or metadata.get("article_id") or "N/A"),
                "content": text,
                "title": title,
            }
        )
    return corpus, records


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u0600-\u06ff\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def simple_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    corpus, records = build_bm25_index()
    q = normalize(query)
    if not q:
        return []
    ranked: list[tuple[float, dict[str, Any]]] = []
    for text, record in zip(corpus, records):
        score = 0.0
        for token in set(q.split()):
            pattern = re.escape(token)
            score += len(re.findall(pattern, normalize(text)))
        if score > 0:
            ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in ranked[:k]]


st.set_page_config(page_title="Legal Intelligence Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Intelligence Engine")
st.caption("Streamlit Community Cloud ready — lightweight legal retrieval demo")

query = st.text_area(
    "اكتب السؤال القانوني",
    height=140,
    placeholder="مثال: ما هي شروط القبض في حالة التلبس؟",
)

if st.button("بحث") and query.strip():
    with st.spinner("جاري البحث في قاعدة النصوص القانونية..."):
        results = simple_search(query, k=5)

    if not results:
        st.warning("لم يتم العثور على نتائج مناسبة. جرّب كلمة أو سؤال مختلفًا.")
    else:
        st.subheader("النتائج")
        for idx, hit in enumerate(results, start=1):
            with st.container():
                st.markdown(f"### {idx}. {hit.get('law_name', 'غير محدد')} / المادة {hit.get('article_id', 'N/A')}")
                st.write(hit.get("content", "")[:500])
                st.caption(hit.get("title", ""))
                st.divider()

st.sidebar.header("About")
st.sidebar.write(
    "هذا التطبيق مصمم ليعمل على Streamlit Community Cloud مع Retrieval خفيف الوزن "
    "بدون تحميل نماذج ثقيلة في وقت التشغيل."
)
st.sidebar.write("يمكن توصيله لاحقًا بواجهة API أو ربط DVC / MLflow / CI في GitHub.")
