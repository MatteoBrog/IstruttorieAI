# istruttorie_ai_app.py

# Patch per usare SQLite moderno su Streamlit Cloud (evita errore Chroma)
try:
    import importlib, sys, pysqlite3
    sys.modules["sqlite3"] = importlib.import_module("pysqlite3")
except ModuleNotFoundError:
    pass

import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import streamlit as st
from asn1crypto import cms

from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.oauth2 import service_account
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# -- Carica variabili in locale, ignorato su Cloud Run/Streamlit Cloud --------
load_dotenv()

# -- Helper per leggere secrets in locale o su Streamlit Cloud -----------------
def get_secret(key: str, default: str | None = None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# -- Parametri da secrets o ambiente -------------------------------------------
API_KEY = os.getenv("API_KEY") or get_secret("API_KEY")
PROJECT_ID = os.getenv("DOC_AI_PROJECT_ID") or get_secret("DOC_AI_PROJECT_ID")
LOCATION = os.getenv("DOC_AI_LOCATION") or get_secret("DOC_AI_LOCATION", "eu")
PROCESSOR_ID = os.getenv("DOC_AI_PROCESSOR_ID") or get_secret("DOC_AI_PROCESSOR_ID")

GEMINI_MODEL_NAME = "gemini-2.0-flash"
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
EMBED_MODEL_NAME = "models/text-embedding-004"

ALLOWED_INPUTS = {".pdf", ".p7m"}

# -- Credenziali Google Service Account ----------------------------------------
try:
    credentials_info = dict(st.secrets["google_service_account"])
    GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_info(credentials_info)
except (StreamlitSecretNotFoundError, KeyError):
    creds_path = "credentials.json"
    if os.path.exists(creds_path):
        GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_file(creds_path)
    else:
        GOOGLE_CREDENTIALS = None

# -- Client Document AI --------------------------------------------------------
if PROJECT_ID and PROCESSOR_ID and GOOGLE_CREDENTIALS is not None:
    _client_opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    DOC_AI_CLIENT = documentai.DocumentProcessorServiceClient(
        credentials=GOOGLE_CREDENTIALS, client_options=_client_opts
    )
else:
    DOC_AI_CLIENT = None

# -- Cache resources (LLM, embeddings, splitter) -------------------------------
@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL_NAME,
        google_api_key=API_KEY,
    )

@st.cache_resource
def get_chat_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        google_api_key=API_KEY,
        temperature=0.2,
        max_output_tokens=2048,
    )

# -- PDF / P7M Helpers ---------------------------------------------------------
def _extract_pdf_text(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return "\n".join(page.get_text() for page in pdf)

def pdf_from_p7m_bytes(data: bytes) -> bytes | None:
    try:
        pkcs7 = cms.ContentInfo.load(data)
        return pkcs7["content"]["encap_content_info"]["content"].native
    except Exception:
        return None

def ocr_pdf(pdf_bytes: bytes) -> str:
    if DOC_AI_CLIENT is None:
        return ""
    raw = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    res = DOC_AI_CLIENT.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw))
    return (res.document.text or "").strip()

# -- Check leggibilità Gemini --------------------------------------------------
_READABILITY_PROMPT = """
Valuta se il testo fornito è leggibile per un essere umano. Rispondi SOLO con una delle due parole:
- leggibile → se il testo contiene parole scritte con caratteri corretti, è comprensibile, ha senso logico e grammaticale.
- illeggibile → se il testo contiene caratteri confusi, simboli casuali, lettere spezzate o non ha senso logico.

Esempio 1:
TESTO:
La presente domanda è presentata ai sensi del bando regionale per il sostegno alle attività agricole.
Risposta: leggibile

Esempio 2:
TESTO:
.R1aò-l.r cEvutFE  
ò:: nL1fA i t#lAF1ìA F-FÉ,A1FIrì  
ioi1eiu Pi';';LA.,i,r . l tuilvrra T:;riae:a 
Risposta: illeggibile

Ora valuta il seguente testo:

TESTO:
"""

def _normalize(ans: str) -> str:
    cleaned = ans.strip().lower()
    if cleaned == "leggibile":
        return "leggibile"
    if cleaned == "illeggibile":
        return "illeggibile"
    return "errore"

def is_readable(sample: str) -> bool:
    if not sample.strip():
        return False
    try:
        llm = get_chat_llm()
        reply = llm.invoke(_READABILITY_PROMPT + sample)
        return _normalize(reply.content) == "leggibile"
    except Exception:
        return False

# -- RAG Pipeline Builder ------------------------------------------------------
def build_rag_pipeline(uploaded: List[st.runtime.uploaded_file_manager.UploadedFile]):
    import shutil
    shutil.rmtree("/tmp/chroma_store", ignore_errors=True)

    docs = []
    processed_names = []

    for up in uploaded:
        name = up.name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_INPUTS:
            st.warning(f"⚠️ {name}: formato non supportato.")
            continue

        data = up.read()
        pdf_bytes = data if suffix == ".pdf" else pdf_from_p7m_bytes(data) or b""
        if not pdf_bytes:
            st.warning(f"⚠️ {name}: errore nell‟estrazione PDF dal P7M.")
            continue

        raw_text = _extract_pdf_text(pdf_bytes)
        sample = " ".join(raw_text.split()[:200])
        readable = is_readable(sample)

        if not readable:
            st.info(f"{name}: testo non leggibile, uso OCR.")
            raw_text = ocr_pdf(pdf_bytes)
            sample = " ".join(raw_text.split()[:200])
            readable = is_readable(sample)

        if readable and raw_text.strip():
            docs.append(Document(page_content=raw_text, metadata={"source": name}))
            processed_names.append(name)
            st.info(f"✅ {name}: indicizzato.")
        else:
            st.warning(f"❌ {name}: illeggibile dopo OCR – scartato.")

    if not docs:
        raise ValueError("Nessun documento utilizzabile.")

    chunks = get_text_splitter().split_documents(docs)

    # ⬇️ Nuovo codice sicuro per Streamlit Cloud
    import shutil
    import tempfile

    # Crea una directory temporanea scrivibile
    persist_dir = tempfile.mkdtemp(prefix="chroma_store_", dir="/tmp")

    # Pulisci prima, se già esisteva qualcosa
    shutil.rmtree(persist_dir, ignore_errors=True)

    # Salva il nuovo database
    vectordb = Chroma.from_documents(chunks, get_embeddings(), persist_directory=persist_dir)
    vectordb.persist()
    retriever = vectordb.as_retriever(k=8)

    return retriever, processed_names, chunks

# -- Domande preimpostate ------------------------------------------------------
PRESET_QUESTIONS = {
    "Verificare la presenza di un Allegato B": "Sono presenti contemporaneamente la dicitura 'Allegato B' e il titolo 'DOMANDA DI PARTECIPAZIONE' nel documento?",
    "Associazione biologica operativa in ≥10 regioni": "riportami dalla Descrizione del soggetto proponente se l'azienda/l'associazione opera in almeno 10 regioni. Indica l'elenco delle regioni individuate con relativo estratto (cita la fonte tra [] con il nome file).",
    "Attività distribuite in ≥5 regioni": "Le attività/interventi del progetto sono distribuiti su almeno 5 regioni italiane...",
    "Verificare la presenza di un Allegato C": "Sono presenti contemporaneamente la dicitura 'Allegato C' e il titolo 'DICHIARAZIONE SOGGETTO PROPONENTE' nel documento?",
    "Verificare la presenza di un Allegato C1": "Sono presenti contemporaneamente la dicitura 'Allegato C1' e il titolo 'DESCRIZIONE PROGETTO' nel documento?",
    "Elenco costi ammissibili ≤50% per regione": "È presente nell'Allegato C1 ... per nessuna regione tali costi superano il 50% del totale...",
    "Rispetto limiti budget 100–500k€": "Il budget totale del progetto... è compreso tra 100000€ e 500000€...",
    "Durata progetto ≤18 mesi": "Verifica se nel documento è indicata la durata... ",
    "Tipologia aiuto e importo finanziamento": "Nell'Allegato C1 ... è indicata la tipologia di aiuto richiesta..."
}

# -- Q&A ------------------------------------------------------------------------
SYSTEM_MSG = (
    "Sei un assistente legale italiano. "
    "Rispondi solo se le informazioni sono chiaramente presenti nel contesto. "
    "Se non trovi la risposta, scrivi 'Non trovato nei documenti'. "
    "Scrivi in italiano chiaro, corretto e professionale. "
    "Cita sempre la fonte tra parentesi quadre, ad esempio [documento.pdf]."

)

def ask(question: str, retriever, llm):
    relevant = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in relevant)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("human",
         "Usa SOLO le informazioni contenute nel contesto per rispondere alla domanda. "
         "Se la risposta non è presente, scrivi 'Non trovato nei documenti'. "
         "Cita sempre la fonte tra parentesi quadre, ad esempio [documento.pdf].\n\n"
         "Domanda: {question}\n\nCONTESTO:\n{context}")
    ])
    reply = llm.invoke(prompt.format(question=question, context=context)).content.strip()
    if not reply:
        return "❌ Nessuna risposta trovata."
    return reply

# -- STREAMLIT UI --------------------------------------------------------------
st.set_page_config(page_title="Istruttorie AI", page_icon="📑")
st.title("📑 Documenti AI – MB demo")

uploaded_files = st.file_uploader("Carica documenti PDF o P7M", type=["pdf", "p7m"], accept_multiple_files=True)

# Reset della sessione solo se i file caricati cambiano
if uploaded_files:
    uploaded_names = sorted([f.name for f in uploaded_files])
    prev_names = st.session_state.get("uploaded_names")

    if uploaded_names != prev_names:
        st.session_state.pop("retriever", None)
        st.session_state.pop("processed_names", None)
        st.session_state.pop("chunks", None)
        st.session_state["uploaded_names"] = uploaded_names

if uploaded_files and st.button("🔄 Indicizza documenti"):
    with st.spinner("⏳ Indicizzazione in corso…"):
        try:
            # Estendi build_rag_pipeline per restituire anche chunks
            retriever, processed, chunks = build_rag_pipeline(uploaded_files)
            st.session_state["retriever"] = retriever
            st.session_state["processed_names"] = processed
            st.session_state["chunks"] = chunks
            st.success("Indicizzazione completata!")
        except Exception as exc:
            st.error(str(exc))

# Mostra lista documenti indicizzati e preview contenuti
if "retriever" in st.session_state:
    retriever = st.session_state["retriever"]

    with st.expander("📑 Documenti indicizzati"):
        st.write("\n".join(st.session_state.get("processed_names", [])))

    with st.expander("🔍 Anteprima contenuti indicizzati"):
        seen_sources = set()
        for d in st.session_state.get("chunks", []):
            src = d.metadata['source']
            if src not in seen_sources:
                st.markdown(f"**Fonte:** {src}**\n\n```{d.page_content[:500]}```")
                seen_sources.add(src)

    with st.expander("✅ Verifiche automatiche", expanded=True):
        run_all = st.button("Esegui tutte le verifiche", key="run_all")
        cols = st.columns(2)
        for idx, (label, question) in enumerate(PRESET_QUESTIONS.items()):
            col = cols[idx % 2]
            if run_all or col.button(label, key=f"btn_{idx}"):
                with st.spinner("Analisi…"):
                    ans = ask(question, retriever, get_chat_llm())
                check = "✅" if ans.lower().startswith("s") else "❌"
                col.markdown(f"{check} **{label}**  \n{ans}")

    prompt = st.text_input("✍️ Fai una domanda…")
    if prompt:
        with st.spinner("💡 Generazione risposta…"):
            st.markdown(ask(prompt, retriever, get_chat_llm()))
else:
    st.info("Carica uno o più documenti e clicca su 'Indicizza documenti' per iniziare.")


st.caption("©2025 – BMTI ScpA.")


#streamlit run app/rag_app_upload.py
#git add requirements.txt app/rag_app_upload.py
#git commit -m "Aggiorna istruttorie app e requirements""
#git push