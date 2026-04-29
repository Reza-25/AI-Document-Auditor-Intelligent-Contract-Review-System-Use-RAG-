import streamlit as st
import os
import time
import tempfile
from rag_engine import ingest_document, query_document, generate_suggested_questions
from audit_rules import AUDIT_CHECKLIST

st.set_page_config(
    page_title="AI Document Auditor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Document Auditor")
st.caption("Upload your contract or document — AI will read, analyze, and answer your questions.")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        # Only re-process if it's a new file
        if st.session_state.get("filename") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("📖 Reading & embedding document..."):
                result = ingest_document(tmp_path)

            st.success(result)
            st.session_state["doc_loaded"] = True
            st.session_state["filename"] = uploaded_file.name
            st.session_state["tmp_path"] = tmp_path
            st.session_state["messages"] = []

            # Generate suggested questions using Groq (fast!)
            with st.spinner("💡 Generating suggested questions..."):
                suggested = generate_suggested_questions(tmp_path)
                st.session_state["suggested_questions"] = suggested
        else:
            st.success(f"✅ Loaded: {uploaded_file.name}")

    st.divider()
    st.markdown("**Stack:**")
    st.caption("🔵 Embedding: Gemini (Google)")
    st.caption("🟠 LLM: Llama 3.3 via Groq (fast!)")
    st.caption("🗄️ Vector DB: ChromaDB (local)")

# ── Main ──────────────────────────────────────────────────────────
if st.session_state.get("doc_loaded"):
    tab1, tab2 = st.tabs(["💬 Ask Questions", "🔍 Auto Audit"])

    # ── Tab 1: Chat ──
    with tab1:
        suggested = st.session_state.get("suggested_questions", [])
        if suggested:
            st.markdown("#### 💡 Suggested Questions")
            st.caption("Click any question to ask instantly:")
            cols = st.columns(2)
            for idx, q in enumerate(suggested):
                if cols[idx % 2].button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state["prefilled_question"] = q
            st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("pages"):
                    st.caption(f"📍 Source: Page(s) {', '.join(map(str, msg['pages']))}")

        prefilled = st.session_state.pop("prefilled_question", None)
        prompt = st.chat_input("Ask anything about the document...") or prefilled

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching & generating answer..."):
                    response = query_document(prompt)
                st.write(response["answer"])
                if response["source_pages"]:
                    st.caption(f"📍 Source: Page(s) {', '.join(map(str, response['source_pages']))}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "pages": response["source_pages"]
            })
            st.rerun()

    # ── Tab 2: Auto Audit ──
    with tab2:
        st.subheader(f"📋 Audit Report: {st.session_state.get('filename', '')}")
        st.info("Groq LLM answers instantly. Only Gemini embedding (1 call/question) may have slight delays.")

        if st.button("🚀 Run Full Audit", type="primary"):
            total_q = sum(len(q) for q in AUDIT_CHECKLIST.values())
            progress = st.progress(0, text="Starting audit...")
            done = 0

            for category, questions in AUDIT_CHECKLIST.items():
                with st.expander(category, expanded=True):
                    for question in questions:
                        progress.progress(done / total_q, text=f"Checking: {question[:55]}...")

                        for attempt in range(3):
                            try:
                                result = query_document(question)
                                break
                            except Exception as e:
                                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 2:
                                    st.warning("Embedding rate limit hit — waiting 20s...")
                                    time.sleep(20)
                                else:
                                    result = {
                                        "answer": f"⚠️ Error: {str(e)[:120]}",
                                        "source_pages": []
                                    }
                                    break

                        st.markdown(f"**Q:** {question}")
                        st.markdown(f"**A:** {result['answer']}")
                        if result["source_pages"]:
                            st.caption(f"📍 Page(s): {', '.join(map(str, result['source_pages']))}")
                        st.divider()

                        done += 1
                        # Small delay to avoid Gemini embedding rate limit
                        if done < total_q:
                            time.sleep(3)

            progress.progress(1.0, text="✅ Audit complete!")
            st.success("🎉 Full audit completed!")

else:
    st.info("👈 Upload a PDF document in the sidebar to get started.")
    st.markdown("""
    ### What this tool can do:
    - 📄 **Read any PDF** — contracts, SOPs, agreements, policies
    - 💡 **Auto-suggest relevant questions** based on your document content
    - 🔍 **Answer questions** with exact page citations
    - 📋 **Run full audit** — checks penalties, confidentiality, termination, jurisdiction & more
    - ⚡ **Powered by Groq** — answers in under 2 seconds
    """)