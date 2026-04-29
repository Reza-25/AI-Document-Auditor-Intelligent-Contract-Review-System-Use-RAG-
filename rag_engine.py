import os
import time
import shutil
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VECTORSTORE_DIR = "./vectorstore"
EMBED_MODEL = "models/gemini-embedding-001"   # Gemini: only for embedding (few API calls)
LLM_MODEL = "llama-3.3-70b-versatile"         # Groq: fast & generous free tier for generation
BATCH_SIZE = 15


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text per page from PDF"""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def ingest_document(pdf_path: str, collection_name: str = "contract") -> str:
    """Pipeline: Extract -> Chunk -> Embed (Gemini) -> Store (ChromaDB)"""

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return "❌ No readable text found in this PDF."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = []
    metadatas = []
    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for chunk in chunks:
            docs.append(chunk)
            metadatas.append({"page": page_data["page"], "source": pdf_path})

    # Gemini embedding — only used here during ingestion (few calls total)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Clear old vectorstore safely (handles Windows file lock)
    if os.path.exists(VECTORSTORE_DIR):
        try:
            shutil.rmtree(VECTORSTORE_DIR)
        except PermissionError:
            import uuid
            collection_name = f"contract_{uuid.uuid4().hex[:8]}"

    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_texts(
                        texts=batch_docs,
                        embedding=embeddings,
                        metadatas=batch_meta,
                        collection_name=collection_name,
                        persist_directory=VECTORSTORE_DIR
                    )
                else:
                    vectorstore.add_texts(texts=batch_docs, metadatas=batch_meta)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = (attempt + 1) * 15
                    print(f"Embedding rate limit on batch {batch_num}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise e

        if i + BATCH_SIZE < len(docs):
            time.sleep(2)

    return f"✅ Done! Processed {len(docs)} chunks from {len(pages)} pages."


def generate_suggested_questions(pdf_path: str) -> list[str]:
    """Use Groq LLM to generate 5 relevant audit questions from document"""
    try:
        pages = extract_text_from_pdf(pdf_path)
        sample_text = "\n".join([p["text"] for p in pages[:3]])

        # Groq — very fast, no rate limit issue for single LLM calls
        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )

        prompt = f"""Based on this document excerpt, generate exactly 5 specific and relevant questions 
that an auditor or legal reviewer would want to ask about this document.
Return ONLY the 5 questions, one per line, numbered 1-5. No explanations, no preamble.

Document excerpt:
{sample_text[:2000]}

5 audit questions:"""

        response = llm.invoke(prompt)
        raw = response.content.strip()

        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                clean = line.lstrip("0123456789.)- ").strip()
                if clean:
                    questions.append(clean)

        return questions[:5] if questions else get_default_questions()

    except Exception as e:
        print(f"Could not generate suggested questions: {e}")
        return get_default_questions()


def get_default_questions() -> list[str]:
    """Fallback questions if generation fails"""
    return [
        "What are the main obligations of each party in this contract?",
        "Are there any penalty or fine clauses? What are the amounts?",
        "What is the duration and renewal process of this agreement?",
        "How can this contract be terminated early?",
        "Are there any confidentiality or non-compete clauses?"
    ]


def query_document(question: str, collection_name: str = "contract") -> dict:
    """Retrieve relevant chunks (Gemini embed) then generate answer (Groq LLM)"""

    # Gemini embedding for query — only 1 API call per question
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""You are a professional document auditor assistant.
Answer the question based ONLY on the context below.
If the answer is not in the context, say "Information not found in the document."
Always cite the page number at the end of your answer, e.g. [Source: Page 3].

Context:
{context}

Question: {question}

Answer:""")

    # Groq LLM — fast generation, generous free tier
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    source_pages = sorted(set(
        doc.metadata["page"]
        for doc in retrieved_docs
        if "page" in doc.metadata
    ))

    return {
        "answer": answer,
        "source_pages": source_pages
    }