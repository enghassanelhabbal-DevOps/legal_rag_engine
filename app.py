from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("LEGAL_API_URL", "")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --card: #1f2937;
            --line: rgba(148, 163, 184, 0.2);
            --accent: #38bdf8;
            --accent-2: #a78bfa;
            --text: #e5eefb;
            --muted: #a8b3c7;
            --success: #34d399;
        }
        .stApp {
            background: linear-gradient(135deg, #020817 0%, #101827 40%, #111827 100%);
            color: var(--text);
        }
        [data-testid="block-container"] {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.12), rgba(167, 139, 250, 0.12));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.4rem 1.3rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.25);
        }
        .result-card {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.2);
        }
        .result-card h3 {
            margin-top: 0;
            margin-bottom: 0.35rem;
            color: #f8fafc;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.14);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.35);
            font-size: 0.76rem;
            margin-bottom: 0.8rem;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .stButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: #08111d;
            font-weight: 700;
            padding: 0.65rem 1rem;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.35);
            color: var(--text);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        {
            "document_id": "sample-3",
            "law_name": "قانون الأحوال الشخصية",
            "article_id": "3",
            "content": "في المعاملات والحقوق الشخصية يراعى المبدأ العام للعدالة والوضوح في التفسير القانوني.",
            "metadata": {"title": "المادة 3 - قانون الأحوال الشخصية"},
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


def render_results(results: list[dict[str, Any]]) -> None:
    st.subheader("نتائج الاسترجاع")
    for idx, hit in enumerate(results, start=1):
        law_name = hit.get("law_name", "غير محدد")
        article_id = hit.get("article_id", "N/A")
        content = hit.get("content", "")
        title = hit.get("title", "")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="badge">Result #{idx}</div>
                <h3>{law_name} / المادة {article_id}</h3>
                <div class="small-muted">{title}</div>
                <p>{content[:500]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    apply_theme()
    st.set_page_config(
        page_title="Legal Intelligence Engine",
        page_icon="⚖️",
        layout="wide",
    )

    st.markdown(
        """
        <div class="hero">
            <h1 style='margin-bottom:0.15rem;'>⚖️ Legal Intelligence Engine</h1>
            <p style='margin:0;color:#a8b3c7;'>Search legal knowledge with a clean, production-ready interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    docs = load_legal_documents()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        """
        <div class='metric-card'>
            <div class='small-muted'>Documents</div>
            <h2>{}</h2>
        </div>
        """.format(len(docs)),
        unsafe_allow_html=True,
    )
    c2.markdown(
        """
        <div class='metric-card'>
            <div class='small-muted'>Mode</div>
            <h2>Hybrid</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        """
        <div class='metric-card'>
            <div class='small-muted'>Status</div>
            <h2>Live</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c4.markdown(
        """
        <div class='metric-card'>
            <div class='small-muted'>Deploy</div>
            <h2>Cloud</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "اكتب السؤال القانوني",
        height=140,
        placeholder="مثال: ما هي شروط القبض في حالة التلبس؟",
    )

    action_col, status_col = st.columns([1, 4])
    do_search = action_col.button("بحث")
    if API_URL:
        status_col.caption(f"API mode: {API_URL}")
    else:
        status_col.caption("Local demo mode: no external API connected")

    if do_search and query.strip():
        with st.spinner("جاري البحث في المحتوى القانوني..."):
            results: list[dict[str, Any]] = []
            if API_URL:
                try:
                    response = requests.post(
                        API_URL,
                        json={"query": query, "top_k": 5},
                        timeout=20,
                    )
                    if response.ok:
                        payload = response.json()
                        results = payload.get("sources", [])
                        if results:
                            st.success("تم جلب النتائج من واجهة API")
                            if payload.get("answer"):
                                st.subheader("إجابة مختصرة")
                                st.write(payload["answer"])
                    else:
                        st.warning("فشل الاتصال بالـAPI؛ سيتم استخدام البحث المحلي كبديل")
                except requests.RequestException:
                    st.warning("واجهة API غير متاحة؛ سيتم استخدام البحث المحلي")
            if not results:
                results = simple_search(query, k=5)
        render_results(results)

    st.sidebar.header("Project overview")
    st.sidebar.write(
        "هذا المشروع يجمع بين Retrieval و Evidence و Quality Gates و CI/CD،\n"
        "مع واجهة مستخدم أنيقة جاهزة للنشر على Streamlit Community Cloud."
    )
    st.sidebar.markdown(
        """
        - DVC tracking
        - Docker + Compose
        - GitHub Actions
        - Regression tests
        - Monitoring / Prometheus
        """
    )


if __name__ == "__main__":
    main()
