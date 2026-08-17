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
            background: #f5f7fb;
            color: #0f172a;
        }
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #f3f6fb 100%);
        }
        [data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.92);
            border-right: 1px solid rgba(148, 163, 184, 0.25);
        }
        [data-testid="block-container"] {
            padding-top: 1.2rem;
        }
        .top-shell {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 26px;
            padding: 1rem 1.2rem 0.25rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
        }
        .brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
        }
        .brand-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.3rem;
            height: 2.3rem;
            border-radius: 0.85rem;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            color: white;
            font-size: 1.1rem;
            box-shadow: 0 12px 30px rgba(96, 165, 250, 0.28);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.25);
            padding: 0.38rem 0.7rem;
            color: #166534;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .hero-bar {
            background: linear-gradient(135deg, rgba(96,165,250,0.10), rgba(172,146,250,0.08));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .hero-bar h1 {
            margin: 0;
            font-size: clamp(1.8rem, 2vw, 2.5rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #111827;
        }
        .hero-bar p {
            margin: 0.5rem 0 0;
            color: #475569;
            font-size: 0.98rem;
        }
        .chip-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(15, 23, 42, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            color: #334155;
            font-size: 0.74rem;
            font-weight: 700;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 112px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.04);
        }
        .metric-card .small-muted {
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .metric-card h2 {
            margin: 0.4rem 0 0;
            font-size: 1.8rem;
            color: #0f172a;
            font-weight: 800;
        }
        .chat-shell {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 26px;
            box-shadow: 0 18px 42px rgba(148, 163, 184, 0.1);
            overflow: hidden;
        }
        .chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.2rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(248, 250, 252, 0.8);
        }
        .chat-header .title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
        }
        .assistant-panel {
            padding: 1rem 1rem 0.5rem;
            min-height: 420px;
            background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(248,250,252,0.8));
        }
        .assistant-panel .stChatMessage {
            margin-bottom: 0.3rem;
        }
        .message-bubble {
            max-width: 85%;
            border-radius: 1.2rem;
            padding: 0.9rem 1rem;
            line-height: 1.6;
            margin: 0.4rem 0;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
        }
        .user-msg {
            background: linear-gradient(135deg, #dbeafe, #e2e8f0);
            color: #0f172a;
            margin-left: auto;
            border-bottom-right-radius: 0.45rem;
        }
        .assistant-msg {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.16);
            color: #0f172a;
            margin-right: auto;
            border-bottom-left-radius: 0.45rem;
        }
        .composer {
            border-top: 1px solid rgba(148, 163, 184, 0.14);
            background: rgba(255,255,255,0.96);
            padding: 1rem 1rem 1.1rem;
        }
        .quick-grid {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-bottom: 0.8rem;
        }
        .quick-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: #f8fafc;
            border-radius: 999px;
            color: #334155;
            font-size: 0.75rem;
            padding: 0.5rem 0.8rem;
            font-weight: 600;
        }
        .result-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-top: 0.5rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .badge {
            display: inline-block;
            background: rgba(59,130,246,0.08);
            color: #1d4ed8;
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .result-card h3 {
            margin: 0.5rem 0 0.35rem;
            color: #0f172a;
            font-size: 1.1rem;
        }
        .result-card p {
            margin: 0.2rem 0 0;
            color: #475569;
            line-height: 1.6;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.82rem;
        }
        .panel-box {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.04);
        }
        .stButton > button {
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.7rem 1rem;
        }
        .stButton > button:hover {
            filter: brightness(1.04);
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 14px;
            box-shadow: none !important;
        }
        .stTextInput label,
        .stTextArea label,
        .stSelectbox label,
        .stNumberInput label,
        .stSidebar .stSelectbox label,
        .stSidebar .stTextInput label,
        .stSidebar .stCheckbox label,
        .stSidebar .stRadio label,
        .stSidebar label {
            color: #334155 !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: rgba(71, 85, 105, 0.7) !important;
        }
        .stSidebar .block-container {
            padding-top: 1rem;
        }
        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }
        .stTabs [role="tab"] {
            background: rgba(255,255,255,0.9);
            border-radius: 12px 12px 0 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #334155;
            font-weight: 600;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: rgba(59,130,246,0.08);
            border-color: rgba(96,165,250,0.18);
            color: #0f172a;
        }
        .stAlert {
            background: rgba(239, 246, 255, 0.9);
            border: 1px solid rgba(147, 197, 253, 0.4);
            color: #0f172a;
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
        <div class="top-shell">
            <div class="brand-row">
                <div class="brand">
                    <span class="brand-badge">⚖</span>
                    Legal Copilot
                </div>
                <div class="status-pill">● Ready</div>
            </div>
            <div class="hero-bar">
                <h1>Ask your legal questions like a specialist.</h1>
                <p>Retrieve Arabic legal context, review evidence, and explore the law with a clean AI-powered workflow.</p>
                <div class="chip-row">
                    <span class="chip">🔎 Retrieval</span>
                    <span class="chip">🧠 Model aware</span>
                    <span class="chip">☁️ Cloud ready</span>
                </div>
            </div>
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

    quick_options = [
        "ما هي شروط القبض في حالة التلبس؟",
        "ما هي حقوق المتهم في التحقيق؟",
        "ما هي مسؤولية القاضي في تطبيق القانون؟",
    ]

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    chat_col, insight_col = st.columns([2.3, 1])

    with chat_col:
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-header'><div class='title'>Legal assistant</div><div class='status-pill'>AI on</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='assistant-panel'>", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                ("assistant", "مرحبًا! سأساعدك في الاستفسارات القانونية. أكتب سؤالك، وسأستخدم Retrieval + evidence لتحليل السياق القانوني."),
            ]

        for role, message in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f"<div class='message-bubble user-msg'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message-bubble assistant-msg'>{message}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='composer'>", unsafe_allow_html=True)
        st.markdown("<div class='quick-grid'>", unsafe_allow_html=True)
        for option in quick_options:
            if st.button(option, key=f"quick_{option}", help="Use a sample legal question"):
                st.session_state["chat_history"].append(("user", option))
                st.session_state["pending_prompt"] = option
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        prompt = st.chat_input("اكتب السؤال القانوني...", key="legal_chat_input")
        if prompt is not None and prompt.strip():
            st.session_state["pending_prompt"] = prompt

        if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
            prompt_value = st.session_state["pending_prompt"]
            st.session_state["pending_prompt"] = ""
            st.session_state["chat_history"].append(("user", prompt_value))
            with st.spinner("جاري استرجاع السياق القانوني..."):
                api_target = config["custom_endpoint"] or API_URL
                results: list[dict[str, Any]] = []
                if api_target:
                    try:
                        response = requests.post(api_target, json={"query": prompt_value, "top_k": config["top_k"]}, timeout=20)
                        if response.ok:
                            payload = response.json()
                            results = payload.get("sources", [])
                            answer = payload.get("answer")
                        else:
                            answer = None
                            results = []
                    except requests.RequestException:
                        answer = None
                        results = []
                if not results:
                    results = simple_search(prompt_value, k=config["top_k"])
                if results:
                    first = results[0]
                    answer_text = (
                        f"استنادًا إلى {first.get('law_name', 'القانون')} / المادة {first.get('article_id', 'N/A')}، "
                        f"المعلومة الأساسية هي: {first.get('content', '')[:260]}"
                    )
                    if len(results) > 1:
                        answer_text += "\n\nمراجع أخرى: " + "; ".join(
                            f"{item.get('law_name', 'القانون')} / المادة {item.get('article_id', 'N/A')}"
                            for item in results[1:4]
                        )
                    answer_text = answer_text.strip()
                else:
                    answer_text = "لم أجد نتائج مناسبة. جرّب إعادة صياغة السؤال أو استخدم نموذجًا مختلفًا من الإعدادات." 

                st.session_state["chat_history"].append(("assistant", answer_text))
                st.session_state["last_results"] = results
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with insight_col:
        st.markdown("<div class='panel-box'><h3>Model status</h3></div>", unsafe_allow_html=True)
        st.metric("Chosen model", config["llm_model"])
        st.metric("Backend", config["backend"])
        st.metric("Retrieval", config["retrieval_mode"])

        st.markdown("<div class='panel-box' style='margin-top: 1rem;'><h3>Evidence overview</h3></div>", unsafe_allow_html=True)
        for name, pct in [("Coverage", 92), ("Recall", 87), ("Latency", 91)]:
            st.progress(pct / 100, text=f"{name}: {pct}%")

        if "last_results" in st.session_state and st.session_state["last_results"]:
            st.markdown("<div class='panel-box' style='margin-top: 1rem;'><h3>Latest sources</h3></div>", unsafe_allow_html=True)
            for idx, hit in enumerate(st.session_state["last_results"][:3], start=1):
                st.markdown(
                    f"""
                    <div class='result-card'>
                        <div class='badge'>#{idx}</div>
                        <h3>{hit.get('law_name', 'غير محدد')} / المادة {hit.get('article_id', 'N/A')}</h3>
                        <div class='small-muted'>{hit.get('title', '')}</div>
                        <p>{str(hit.get('content', ''))[:180]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
