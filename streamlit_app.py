"""Nano Voice Agents web UI: upload PDFs, inspect chunk starts, chat via text or audio.

Run:
    1) .\\.venv\\Scripts\\python.exe -m uvicorn api:app --port 8000
    2) .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import httpx
import streamlit as st

API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
st.title("Nano Voice Agents — Support Console 🎙️📄")

# ---------------------------------------------------------------- health
try:
    h = httpx.get(f"{API}/health", timeout=10).json()
    st.caption(f"API ok={h.get('ok')} · points={h.get('points')} · {h.get('collection')}")
except Exception as e:
    st.error(f"API not reachable at {API}: {e}\nStart it: `python -m uvicorn api:app`")
    st.stop()

tab_upload, tab_chunks, tab_chat = st.tabs(["Upload PDFs", "Chunk starts", "Chat (text / audio)"])

# ---------------------------------------------------------------- upload
with tab_upload:
    st.subheader("Upload PDFs → auto-ingest + chunk")
    files = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest uploaded PDFs", disabled=not files):
        prog = st.progress(0)
        for i, f in enumerate(files):
            with st.spinner(f"Uploading {f.name}…"):
                r = httpx.post(
                    f"{API}/documents", files={"file": (f.name, f.getvalue())},
                    timeout=300,
                )
                if r.status_code == 200:
                    res = r.json()
                    st.success(f"{f.name}: {res.get('chunks')} chunks")
                    with st.expander(f"Chunk starts — {f.name}"):
                        for p in (res.get("preview") or [])[:20]:
                            head = " > ".join(p.get("headings") or []) or "(no heading)"
                            st.markdown(
                                f"**#{p['index']}** p.{p.get('page_no')} · `{head}`\n\n"
                                f"> {p.get('start','')[:200]}…"
                            )
                else:
                    st.error(f"{f.name} failed: {r.text[:500]}")
            prog.progress((i + 1) / len(files))
    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = httpx.get(f"{API}/documents", timeout=30).json().get("documents", [])
        for d in docs:
            c1, c2 = st.columns([4, 1])
            c1.write(f"📄 **{d['source']}** — {d['chunks']} chunks")
            if c2.button("Delete", key=f"del-{d['source']}"):
                httpx.delete(f"{API}/documents/{d['source']}", timeout=60)
                st.rerun()
    except Exception as e:
        st.warning(f"Could not list documents: {e}")

# ---------------------------------------------------------------- chunks
with tab_chunks:
    st.subheader("Inspect chunk starts")
    try:
        docs = httpx.get(f"{API}/documents", timeout=30).json().get("documents", [])
        opts = ["(all)"] + [d["source"] for d in docs]
        sel = st.selectbox("Document", opts)
        src = None if sel == "(all)" else sel
        params = {"limit": st.slider("Limit", 10, 500, 100)}
        if src:
            params["source"] = src
        chunks = httpx.get(f"{API}/chunks", params=params, timeout=30).json().get("chunks", [])
        st.caption(f"{len(chunks)} chunks")
        for c in chunks:
            head = " > ".join(c.get("headings") or []) or "(no heading)"
            with st.expander(f"#{c['id'][:8]} · {c.get('source')} · p.{c.get('page_no')} · {head}"):
                st.code(c.get("chunk_text", "")[:2000])
    except Exception as e:
        st.warning(f"Could not load chunks: {e}")

# ---------------------------------------------------------------- chat
with tab_chat:
    st.subheader("Ask — text or voice")
    mode = st.radio("Input", ["Text", "Audio upload"], horizontal=True)
    top_k = st.slider("top_k", 1, 10, 5)
    query = ""
    audio_bytes = None
    if mode == "Text":
        query = st.chat_input("Ask about plans, coverage, balance…") or ""
    else:
        mic = st.audio_input("Record or upload your question")
        if mic is not None:
            audio_bytes = mic.getvalue()
            st.audio(audio_bytes)
            if st.button("Transcribe + Answer"):
                with st.spinner("Voice → text → RAG…"):
                    r = httpx.post(
                        f"{API}/voice", files={"file": ("q.webm", audio_bytes)},
                        params={"top_k": top_k}, timeout=300,
                    )
                    if r.status_code == 200:
                        st.audio(r.content, format="audio/mp3")
                        st.info(f"Transcript: {r.headers.get('X-Transcript','')}")
                        st.success(r.headers.get("X-Answer", "")[:1000])
                    elif r.status_code == 501:
                        st.warning(
                            r.json().get("detail")
                            + " — fallback: type the question in Text mode."
                        )
                    else:
                        st.error(r.text[:1000])

    if "history" not in st.session_state:
        st.session_state.history = []
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    if query:
        st.chat_message("user").write(query)
        with st.spinner("Retrieving…"):
            r = httpx.post(
                f"{API}/chat",
                json={"query": query, "top_k": top_k, "style": "text"},
                timeout=120,
            )
        if r.status_code == 200:
            data = r.json()
            st.chat_message("assistant").write(data["answer"])
            with st.expander("Sources"):
                for s in data.get("sources", []):
                    st.markdown(
                        f"**{s.get('source')}** p.{s.get('page_no')} "
                        f"(score {s.get('score')}) — {s.get('start','')[:200]}…"
                    )
            st.session_state.history += [("user", query), ("assistant", data["answer"])]
        else:
            st.error(r.text[:1000])

    st.divider()
    st.caption("Live voice call (same agent as console): fetch a token then join from a LiveKit web client.")
    if st.button("Get LiveKit token"):
        try:
            t = httpx.get(f"{API}/livekit/token", timeout=15).json()
            st.code(f"URL: {t['url']}\nRoom: {t['room']}\nToken: {t['token'][:60]}…")
        except Exception as e:
            st.warning(f"Token endpoint unavailable: {e}")
