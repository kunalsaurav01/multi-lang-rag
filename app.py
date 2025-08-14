import yaml
import streamlit as st
from dotenv import load_dotenv

from src.utils.lang import detect_lang, pick_ui_lang
from src.translate import translate
from src.retrieve import Retriever
from src.generator import generate_answer, QAInput
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"


load_dotenv()
cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))

st.set_page_config(page_title=cfg["app"]["title"], page_icon="🌍", layout="wide")
st.title(cfg["app"]["title"])
st.write(cfg["app"]["description"])

with st.sidebar:
    st.subheader("Settings")
    ui_lang = pick_ui_lang(st.selectbox(
        "Preferred answer language (ISO code)",
        ["en","hi","es","fr","de","ar","zh","ja","pt","ru"], index=0
    ))
    top_k = st.slider("Top-K passages", 1, 10, cfg["retrieval"]["top_k"])
    st.caption(f"Index: {cfg['data']['persist_dir']}")
    st.info("To (re)build the index, run:\n\n`python -m src.ingest`")

query = st.text_input("Ask anything (any language)")
go = st.button("Search")

if go and query.strip():
    q_lang = detect_lang(query)
    st.write(f"Detected query language: `{q_lang}`")

    retriever = Retriever(cfg["data"]["persist_dir"], cfg["models"]["embedding"], top_k=top_k)
    docs = retriever.search(query)

    if not docs:
        st.warning("No results found. Add multilingual .txt/.md files to `data/` and run the indexer.")
    else:
        st.subheader("Top Passages")
        for i, d in enumerate(docs, start=1):
            m = d["metadata"]
            st.markdown(f"**Doc {i}** — {m.get('source','?')} · lang={m.get('lang','?')} · distance={d['distance']:.4f}")
            st.write(d["text"])

        st.subheader("Answer")
        qa_input = QAInput(
            query=query,
            passages=[{"text": d["text"], "source_lang": d["metadata"].get("lang","en"), "metadata": d["metadata"]} for d in docs]
        )
        raw_answer = generate_answer(qa_input, answer_lang=ui_lang)

        # Safety: ensure the final output is in the user's preferred language
        detected = detect_lang(raw_answer) if raw_answer.strip() else ui_lang
        final_answer = raw_answer if detected == ui_lang else translate(raw_answer, detected, ui_lang)

        st.success(final_answer)

        st.markdown("**Citations**")
        for i, d in enumerate(docs, start=1):
            m = d["metadata"]
            st.write(f"[Doc {i}] {m.get('source','?')} (lang={m.get('lang','?')})")
else:
    st.info("Enter a question and click Search. Put .txt/.md files into `data/`, then run `python -m src.ingest`.")
