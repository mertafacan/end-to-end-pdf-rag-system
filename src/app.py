import os
import requests
import streamlit as st

st.set_page_config(page_title="RAG Assistant", page_icon="💬", layout="centered")
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Initialize session state variables
if "session_id" not in st.session_state: st.session_state.session_id = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "uploads" not in st.session_state: st.session_state.uploads = []
if "processing" not in st.session_state: st.session_state.processing = False

def _preview(text: str, n: int = 5) -> str:
    """Show first n lines of text for preview"""
    text = (text or "").strip()
    lines = text.splitlines()
    return "\n".join(lines[:n])

def show_sources_popover(sources: list):
    """Display sources in a popover with preview and expandable full text"""
    if not sources: return
    pop = st.popover(f"📚 Sources ({len(sources)})")
    with pop:
        for i, src in enumerate(sources, 1):
            label = src.get("label") or "Source"
            page = src.get("page")
            score = src.get("score_raw")
            meta = []
            if page is not None: meta.append(f"page {page}")
            if score is not None:
                try: meta.append(f"score {float(score):.3f}")
                except: pass
            st.markdown(f"**{i}. {label}**" + (f" · _{' · '.join(meta)}_" if meta else ""))
            text = src.get("text") or ""
            st.write(_preview(text, n=5))
            if text and len(text.splitlines()) > 5:
                with st.expander("continue"):
                    st.write(text)
            if i < len(sources): st.divider()

with st.sidebar:
    st.header("📄 Upload PDF")
    st.session_state.uploads = st.file_uploader(
        "Select PDF", type=["pdf"], accept_multiple_files=True
    ) or []
    if st.button("📥 Index", disabled=not st.session_state.uploads, use_container_width=True):
        try:
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in st.session_state.uploads]
            with st.spinner("Indexing…"):
                r = requests.post(f"{API_URL}/index", files=files, timeout=(10, 120))
                r.raise_for_status()
                resp = r.json()
            if "error" in resp:
                st.error(f"Indexing error: {resp['error']}")
            else:
                st.session_state.session_id = resp.get("session_id", "")
                st.session_state.messages.clear()
                st.toast("PDFs added to index ✅")
        except Exception as e:
            st.error(f"Indexing failed: {e}")
    st.divider()
    if st.button("🗑️ Reset chat", use_container_width=True):
        st.session_state.messages.clear()
        st.toast("Chat cleared")

st.title("💬 RAG Assistant")

if not st.session_state.session_id:
    st.info("👈 To get started, upload PDFs from the left and click **Index**.")
    st.stop()

# Display chat history
for m in st.session_state.messages:
    st.chat_message("user").write(m["q"])
    with st.chat_message("assistant"):
        st.write(m.get("a", ""))
        show_sources_popover(m.get("sources", []))

# Handle new user question
user_q = st.chat_input("Write your question…")
if user_q and user_q.strip() and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({"q": user_q, "a": "", "sources": []})
    
    st.chat_message("user").write(user_q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                r = requests.post(
                    f"{API_URL}/ask",
                    json={"session_id": st.session_state.session_id, "question": user_q},
                    timeout=(10, 120),
                )
                r.raise_for_status()
                resp = r.json()
            if "error" in resp:
                ans, srcs = f"Error: {resp['error']}", []
            else:
                ans, srcs = resp.get("answer", ""), resp.get("sources", [])
            st.write(ans)
            show_sources_popover(srcs)
            # Update the last message with the response
            st.session_state.messages[-1] = {"q": user_q, "a": ans, "sources": srcs}
        except Exception as e:
            err = f"Error: {e}"
            st.error(err)
            st.session_state.messages[-1] = {"q": user_q, "a": err, "sources": []}
        finally:
            st.session_state.processing = False
            st.rerun()