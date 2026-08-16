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

st.set_page_config(
    page_title="Legal Intelligence Engine",
    page_icon="⚖️",
    layout="wide",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #071421;
            color: #edf6ff;
        }
        .stApp {
            background: linear-gradient(135deg, #071421 0%, #0d1728 55%, #0f1d32 100%);
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="block-container"] {
            padding-top: 1.2rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(168,85,247,0.18));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.22);
        }
        .hero h1 {
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #f8fbff;
        }
        .hero p {
            margin: 0.6rem 0 0;
            color: #c3d1e7;
            font-size: 1.05rem;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.16);
        }
        .metric-card .small-muted {
            color: #9db2ce;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-card h2 {
            margin: 0.4rem 0 0;
            font-size: 2rem;
            font-weight: 700;
            color: #f8fbff;
        }
        .result-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.15);
        }
        .result-card h3 {
            margin: 0.2rem 0 0.5rem;
            color: #f2f7ff;
            font-size: 1.18rem;
        }
        .result-card p {
            margin: 0.4rem 0 0;
            color: #dfe9fa;
            line-height: 1.6;
        }
        .badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.13);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.32);
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 600;
        }
        .small-muted {
            color: #a6b7d1;
            font-size: 0.82rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            color: #08111d;
            border: none;
            border-radius: 12px;
            font-weight: 800;
            padding: 0.7rem 1.2rem;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div {
            background: rgba(15, 23, 42, 0.75);
            color: #edf6ff;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
        }
        .stSidebar .stSelectbox label,
        .stSidebar .stTextInput label,
        .stSidebar .stCheckbox label,
        .stSidebar .stRadio label {
            color: #dfe9fa;
        }
        .stSidebar .block-container {
            padding-top: 1rem;
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


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Control center")
        st.caption("Choose your model and API settings")

        llm_model = st.selectbox(
            "LLM model",
            [
                "Qwen 3 (local)",
                "Gemini 2.5",
                "GPT-4o mini",
                "Llama 3.1",
                "Claude Sonnet",
            ],
            index=0,
        )
        backend = st.selectbox(
            "Transformer backend",
            [
                "Transformers",
                "vLLM",
                "LangChain",
                "OpenAI-compatible API",
            ],
            index=0,
        )
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["Hybrid", "Dense", "BM25"],
            index=0,
        )
        top_k = st.slider("Top results", min_value=3, max_value=10, value=5)

        st.markdown("### API keys")
        openai_key = st.text_input("OpenAI API key", type="password", help="Optional for cloud LLM access")
        google_key = st.text_input("Google API key", type="password", help="Optional for Gemini access")
        hf_token = st.text_input("Hugging Face token", type="password", help="Optional for model downloads / private models")
        custom_endpoint = st.text_input(
            "Custom endpoint",
            value="",
            placeholder="https://api.example.com/v1",
        )

        st.markdown("### Project overview")
        st.markdown(
            "- DVC tracking\n- Docker + Compose\n- GitHub Actions\n- Regression tests\n- Monitoring / Prometheus"
        )

        return {
            "llm_model": llm_model,
            "backend": backend,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "openai_key": openai_key,
            "google_key": google_key,
            "hf_token": hf_token,
            "custom_endpoint": custom_endpoint,
        }


def main() -> None:
    apply_theme()

    config = render_sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>⚖️ Legal Intelligence Engine</h1>
            <p>Search legal knowledge with a clean, production-ready interface.</p>
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
            <h2>{}</h2>
        </div>
        """.format(config["retrieval_mode"]),
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

    st.info(
        f"Selected model: {config['llm_model']} | Backend: {config['backend']} | Retrieval: {config['retrieval_mode']}"
    )

    question = st.text_area(
        "اكتب السؤال القانوني",
        height=150,
        placeholder="مثال: ما هي شروط القبض في حالة التلبس؟",
    )

    quick_options = [
        "ما هي شروط القبض في حالة التلبس؟",
        "ما هي حقوق المتهم في التحقيق؟",
        "ما هي مسؤولية القاضي في تطبيق القانون؟",
    ]
    quick = st.columns(len(quick_options))
    for column, option in zip(quick, quick_options):
        if column.button(option, key=option):
            question = option

    search_clicked = st.button("بحث", use_container_width=True)

    if search_clicked and question.strip():
        with st.spinner("جاري البحث في المحتوى القانوني..."):
            results: list[dict[str, Any]] = []
            api_target = config["custom_endpoint"] or API_URL
            if api_target:
                try:
                    response = requests.post(
                        api_target,
                        json={"query": question, "top_k": config["top_k"]},
                        timeout=20,
                    )
                    if response.ok:
                        payload = response.json()
                        results = payload.get("sources", [])
                        if payload.get("answer"):
                            st.subheader("إجابة مختصرة")
                            st.write(payload["answer"])
                            st.success("تم جلب النتائج من واجهة الـAPI")
                    else:
                        st.warning("فشل الاتصال بالـAPI. سيتم استخدام البحث المحلي كبديل.")
                except requests.RequestException:
                    st.warning("واجهة الـAPI غير متاحة. سيتم استخدام البحث المحلي كبديل.")
            if not results:
                results = simple_search(question, k=config["top_k"])
        render_results(results)


if __name__ == "__main__":
    main()
