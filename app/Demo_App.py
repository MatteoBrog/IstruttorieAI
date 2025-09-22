# ─────────────────────────────────────────────────────────────────────────────
# Patch per usare SQLite moderno su Streamlit Cloud (evita errore Chroma/FAISS)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import importlib
    import sys
    import pysqlite3

    sys.modules["sqlite3"] = importlib.import_module("pysqlite3")
except ModuleNotFoundError:
    pass

# ----------------------------- Terze parti (base) ---------------------------
import streamlit as st

# ⚠️ Best practice: set_page_config PRIMA di qualsiasi altro elemento UI
st.set_page_config(page_title="Istruttorie AI", layout="wide")

# ----------------------------- Built-in ------------------------------------
import re
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import string
import hashlib
import unicodedata
from collections import Counter
from datetime import datetime

# ----------------------------- Terze parti ---------------------------------
import fitz  # PyMuPDF
from asn1crypto import cms
from dotenv import load_dotenv
import numpy as np

from streamlit.runtime.secrets import StreamlitSecretNotFoundError

from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography import x509

# Google Cloud – Document AI e autenticazione
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

# LangChain & Vertex AI
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate

# Hybrid search
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

SHOW_VARIANTS = False

# --------------------------------- Setup -----------------------------------
load_dotenv()

PROJECT_ID: Optional[str] = os.getenv("PROJECT_ID") or st.secrets.get("PROJECT_ID")
PROCESSOR_ID: Optional[str] = os.getenv("DOC_AI_PROCESSOR_ID") or st.secrets.get("DOC_AI_PROCESSOR_ID")
DOC_AI_LOCATION: str = os.getenv("DOC_AI_LOCATION") or st.secrets.get("DOC_AI_LOCATION", "eu")
VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION") or st.secrets.get("VERTEX_LOCATION", "europe-west4")

# Credenziali Google Service Account
try:
    credentials_info = dict(st.secrets["google_service_account"])
    GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_info(credentials_info)
except (StreamlitSecretNotFoundError, KeyError):
    if os.path.exists("credentials.json"):
        GOOGLE_CREDENTIALS = service_account.Credentials.from_service_account_file("credentials.json")
    else:
        GOOGLE_CREDENTIALS = None


# -------------------------- Utilities: unique keys --------------------------
def make_key(prefix: str, *parts) -> str:
    """
    Genera chiavi uniche per widget Streamlit evitando collisioni.
    """
    base = "|".join(str(p) for p in parts)
    h = hashlib.md5(base.encode()).hexdigest()[:8]
    c = st.session_state.setdefault("__key_counter__", 0)
    st.session_state["__key_counter__"] = c + 1
    return f"{prefix}_{h}_{c}"


def stable_key(prefix: str, *parts) -> str:
    """
    Chiave *stabile* per i widget che devono ricordare lo stato (es. bottoni che aprono anteprime).
    Non usa contatori, ma solo un hash dei parametri.
    """
    base = "|".join(str(p) for p in parts)
    h = hashlib.md5(base.encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


# ---------------- PDF / P7M extraction & OCR -------------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Estrae testo da un PDF con separatore form feed tra pagine (\f)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pieces = []
        for i in range(pdf.page_count):
            pieces.append(pdf[i].get_text())
            if i != pdf.page_count - 1:
                pieces.append("\f")
        return "".join(pieces)


def extract_pdf_from_p7m(p7m_bytes: bytes) -> Optional[bytes]:
    """Estrae PDF da busta P7M (PKCS#7)."""
    try:
        ci = cms.ContentInfo.load(p7m_bytes)
        return ci["content"]["encap_content_info"]["content"].native
    except Exception:
        return None


def _pdf_subbytes(pdf_bytes: bytes, start_page: int, end_page: int) -> bytes:
    """Restituisce un nuovo PDF bytes con pagine [start_page, end_page] (0-based)."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    dst.insert_pdf(src, from_page=start_page, to_page=end_page)
    out = dst.tobytes()
    dst.close()
    src.close()
    return out


@st.cache_resource(show_spinner=False)
def get_docai_client():
    if not (PROJECT_ID and PROCESSOR_ID and GOOGLE_CREDENTIALS):
        return None
    return documentai.DocumentProcessorServiceClient(
        credentials=GOOGLE_CREDENTIALS,
        client_options=ClientOptions(api_endpoint=f"{DOC_AI_LOCATION}-documentai.googleapis.com"),
    )


# --- helper: iterazione a blocchi di pagine per Document AI (max 15) ---
def _iter_pdf_chunks(pdf_bytes: bytes, chunk_pages: int = 15):
    """
    Spezza il PDF in sotto-PDF da <= chunk_pages e produce tuple:
    (sub_pdf_bytes, start_page_index_0_based, end_page_index_0_based_incluso)
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        n = doc.page_count
        for start in range(0, n, chunk_pages):
            end = min(n - 1, start + chunk_pages - 1)
            yield _pdf_subbytes(pdf_bytes, start, end), start, end


def run_ocr_with_document_ai(pdf_bytes: bytes) -> str:
    """
    Esegue OCR a blocchi (<=15 pagine) e concatena i risultati.
    Ritorna il testo completo separando i blocchi con \f per preservare i confini pagina.
    """
    client = get_docai_client()
    if client is None or not PROJECT_ID or not DOC_AI_LOCATION or not PROCESSOR_ID:
        return ""

    # N.B. name supporta sia processor che processorVersion; qui rimaniamo su processor
    name = f"projects/{PROJECT_ID}/locations/{DOC_AI_LOCATION}/processors/{str(PROCESSOR_ID).strip('.')}"

    out_parts: List[str] = []
    total_pages = 0

    for sub_pdf, start_idx, end_idx in _iter_pdf_chunks(pdf_bytes, chunk_pages=15):
        # fino a 3 tentativi per chunk
        last_err = None
        for attempt in range(3):
            try:
                raw = documentai.RawDocument(content=sub_pdf, mime_type="application/pdf")
                resp = client.process_document(
                    request=documentai.ProcessRequest(name=name, raw_document=raw)
                )
                text = (resp.document.text or "").strip()
                # Aggiungi un form feed tra blocchi per rispettare i confini
                if text:
                    if out_parts:
                        out_parts.append("\f")
                    out_parts.append(text)
                total_pages += (end_idx - start_idx + 1)
                break  # chunk ok
            except Exception as e:
                last_err = e
                if attempt < 2:
                    continue
                # dopo 3 tentativi, salta il chunk ma non bloccare tutto
                # (opzionale: puoi loggare con st.warning)
                # st.warning(f"OCR: fallito chunk pagine {start_idx+1}-{end_idx+1}: {e}")
        # continua con i chunk successivi

    return "".join(out_parts).strip()


def _process_blob(blob: bytes) -> str:
    """
    Versione chunked identica a run_ocr_with_document_ai (riuso per coerenza).
    """
    client = get_docai_client()
    if client is None or not PROJECT_ID or not DOC_AI_LOCATION or not PROCESSOR_ID:
        return ""

    name = f"projects/{PROJECT_ID}/locations/{DOC_AI_LOCATION}/processors/{str(PROCESSOR_ID).strip('.')}"

    out_parts: List[str] = []
    for sub_pdf, start_idx, end_idx in _iter_pdf_chunks(blob, chunk_pages=15):
        last_err = None
        for attempt in range(3):
            try:
                raw = documentai.RawDocument(content=sub_pdf, mime_type="application/pdf")
                resp = client.process_document(
                    request=documentai.ProcessRequest(name=name, raw_document=raw)
                )
                text = (resp.document.text or "").strip()
                if text:
                    if out_parts:
                        out_parts.append("\f")
                    out_parts.append(text)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    continue
                # st.warning(f"OCR: fallito chunk pagine {start_idx+1}-{end_idx+1}: {e}")
    return "".join(out_parts).strip()


def estrai_firmatari(p7m_bytes: bytes) -> List[dict]:
    """Dati firmatari da P7M (se presente)."""
    try:
        certs = pkcs7.load_der_pkcs7_certificates(p7m_bytes)
        out = []
        for cert in certs:
            out.append({
                "CN": cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value,
                "Emesso da": cert.issuer.rfc4514_string(),
                "Valido dal": cert.not_valid_before.isoformat(),
                "Valido fino": cert.not_valid_after.isoformat(),
            })
        return out
    except Exception as e:
        return [{"Errore": str(e)}]


# -------------------- Leggibilità rapida + fallback LLM --------------------
_READABILITY_PROMPT = (
    "Valuta se il testo seguente è leggibile per un essere umano. "
    "Rispondi SOLO con: leggibile / illeggibile.\n\nTESTO:\n"
)


def _normalize(ans: str) -> str:
    cleaned = ans.strip().lower()
    if cleaned == "leggibile":
        return "leggibile"
    if cleaned == "illeggibile":
        return "illeggibile"
    return "errore"


def _quick_legible(sample: str) -> bool:
    if len(sample) < 60:
        return False
    letters = sum(c.isalpha() for c in sample)
    spaces = sample.count(" ")
    punct = sum(c in string.punctuation for c in sample)
    ratio_letters = letters / max(1, len(sample))
    ratio_spaces = spaces / max(1, len(sample))
    return (ratio_letters > 0.55 and ratio_spaces > 0.10 and punct < len(sample) * 0.15)


def is_readable(sample: str) -> bool:
    if not sample.strip():
        return False
    if _quick_legible(sample):
        return True
    try:
        llm = get_llm()
        reply = llm.invoke(_READABILITY_PROMPT + sample)
        return _normalize(reply.content) == "leggibile"
    except Exception:
        return False


# -------------------- Tag “morbidi” (solo boost, no filtri) ----------------
FILENAME_TAGS = {
    "PEC": [re.compile(r"\bpec\b", re.I), re.compile(r"ricevut|accettaz|consegn", re.I)],
    "Allegato_B": [re.compile(r"allegato[_\s-]*b(\b|\.|_)", re.I),
                   re.compile(r"domanda[_\s-]*di[_\s-]*partecipazione", re.I)],
    "Allegato_C": [re.compile(r"allegato[_\s-]*c(\b|\.|_)(?!\s*\d)", re.I)],
    "Allegato_C1": [re.compile(r"allegato[_\s-]*c1(\b|\.|_)", re.I)],
    "Allegato_C2": [re.compile(r"allegato[_\s-]*c2(\b|\.|_)", re.I)],
    "Allegato_C3": [re.compile(r"allegato[_\s-]*c3(\b|\.|_)", re.I)],
    "Allegato_D": [
        re.compile(r"allegato[_\s-]*d(\b|\.|_)", re.I),
        re.compile(r"dichiarazion[ei]\s+soggetto\s+beneficiario", re.I),
    ],
    "Allegato_E": [re.compile(r"allegato[_\s-]*e(\b|\.|_)", re.I),
                   re.compile(r"allegato[_\s-]*eformat", re.I)],

}


def classify_by_filename(filename: str) -> Set[str]:
    tags: Set[str] = set()
    fn = filename.lower()
    for tag, patterns in FILENAME_TAGS.items():
        if any(p.search(fn) for p in patterns):
            tags.add(tag)
    return tags


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I | re.S | re.M)


# Indizi dal CONTENUTO (assegnati come boost)
CONTENT_TAG_HINTS: Dict[str, List[re.Pattern]] = {
    "PEC": [
        _rx(r"\bpec\b|posta elettronica certificat"),
        _rx(r"ricevut[ae]\s+di\s+accettazione|avvenuta\s+consegna|ricevuta di consegna"),
    ],
    "Allegato_B": [
        _rx(r"\ballegato\s*b\b"),
        _rx(r"domanda\s+di\s+partecipazione"),
    ],
    "Allegato_C": [
        _rx(r"\ballegato\s*c(?!\d)\b"),
        _rx(r"dichiarazione\s+soggetto\s+proponente|dichiara ai sensi.*445/2000"),
    ],
    "Allegato_C1": [
        _rx(r"\ballegato\s*c1\b"),
        _rx(r"descrizion[ea]\s+progetto|titolo\s+progetto"),
    ],
    "Allegato_C2": [
        _rx(r"\ballegato\s*c2\b"),
        _rx(r"dettaglio\s+finanziari\w*|attivit[aà]\s+finanziabili"),
    ],
    "Allegato_C3": [
        _rx(r"\ballegato\s*c3\b"),
        _rx(r"ripartizion[ea]|articolo\s+21\s+regolamento\s*\(ue\)\s*2022/2472"),
    ],
    "Allegato_D": [
        _rx(r"\ballegato\s*d\b"),
        _rx(r"dichiarazion[ei]\s+soggetto\s+beneficiario"),
        _rx(r"art\.?\s*46\s+del\s+dpr\s*445/2000"),  # ricorre nella dichiarazione
    ],
    "Allegato_E": [
        _rx(r"allegatoeformat"),
        _rx(r"dichiarazione\s+di\s+impegno\s+alla\s+costituzione\s+della\s+filiera\s+biologica"),
    ],
}


def classify_by_content(text: str) -> Set[str]:
    tags: Set[str] = set()

    # ---- PEC (mantieni robusto): serve "pec" + una ricevuta (accettazione/consegna)
    has_pec = bool(_rx(r"\bpec\b|posta elettronica certificat").search(text))
    has_receipt = bool(_rx(r"ricevut[ae]\s+di\s+accettazione|avvenuta\s+consegna|ricevuta di consegna").search(text))
    if has_pec and has_receipt:
        tags.add("PEC")

    # ---- Allegati: usa le regex "forti" riusate dalla checklist, ma SOLO intestazione
    tags |= detect_attachment_tags(text)

    return tags


# =========================
# ANALISI RICEVIBILITÀ PEC
# =========================

EXPECTED_SUBJECT = (
    "Istanza per la concessione di agevolazioni volte a favorire le forme di produzione agricola a "
    "ridotto impatto ambientale e per la promozione di Filiere e distretti di agricoltura biologica - Filiere biologiche"
)

# Finestra temporale di ricevibilità (inclusiva)
WINDOW_START = datetime(2024, 6, 12, 12, 0, 0)
WINDOW_END = datetime(2024, 6, 26, 12, 0, 0)

_RX_PEC = re.compile(r"\bpec\b|posta\s+elettronica\s+certificat", re.I)
_RX_RECEIPT = re.compile(r"ricevut[ae]\s+di\s+accettazione|avvenuta\s+consegna|ricevuta\s+di\s+consegna", re.I)
_RX_SUBJECT = re.compile(r"(?:^|\n)\s*(?:oggetto|subject)\s*[:\-]\s*(.+)", re.I)
_RX_PROTO = re.compile(r"(?:protocollo|prot\.)\s*[:\-]?\s*([A-Z0-9\/\.\-]{4,})", re.I)
_RX_SENDER_LINE = re.compile(
    r"(?:^|\n)\s*(?:mittente|from|da|inviato da)\s*[:\-]\s*(.+)", re.I
)
_RX_EMAIL = re.compile(
    r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I
)

# Date e ore più comuni nelle ricevute PEC italiane
_DATE_PATTERNS = [
    r"(\d{2}/\d{2}/\d{4})\s+(?:alle\s+ore\s+)?(\d{2}[:\.]\d{2}(?::\d{2})?)",
    r"(?:inviat[ao] il|spedita il|consegnat[ao] il)\s+(\d{2}/\d{2}/\d{4})\s+(?:alle\s+ore\s+)?(\d{2}[:\.]\d{2}(?::\d{2})?)",
    r"(?:del)\s+(\d{2}/\d{2}/\d{4})\s+(?:alle\s+ore\s+)?(\d{2}[:\.]\d{2}(?::\d{2})?)",
]
_RX_DATES = [re.compile(p, re.I) for p in _DATE_PATTERNS]


def _norm_time(t: str) -> str:
    # Converte "12.00" -> "12:00"
    return t.replace(".", ":")


def _parse_dt(date_s: str, time_s: str) -> Optional[datetime]:
    try:
        time_s = _norm_time(time_s)
        fmts = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
        for f in fmts:
            try:
                return datetime.strptime(f"{date_s} {time_s}", f)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _first_dt_in_text(text: str) -> Optional[datetime]:
    for rx in _RX_DATES:
        m = rx.search(text)
        if m:
            dt = _parse_dt(m.group(1), m.group(2))
            if dt:
                return dt
    return None


def _find_on_page(text_ff: str, rx: re.Pattern) -> Tuple[Optional[str], Optional[int]]:
    m = rx.search(text_ff)
    if not m:
        return None, None
    quote = re.sub(r"\s+", " ", (m.group(0) if m.lastindex is None else m.group(m.lastindex))).strip()[:240]
    page = _page_of_match(text_ff, m.start())
    return quote, page


def _find_sender(text_ff: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Ritorna (sender_display, quote, page).
    - Prima prova con riga 'Mittente/From/Da/Inviato da: ...'
    - Poi fallback: prima email trovata vicino a parole chiave PEC.
    """
    # 2.1 — preferisci la prima occorrenza con etichetta
    m = _RX_SENDER_LINE.search(text_ff)
    if m:
        raw = m.group(1).strip()
        email_m = _RX_EMAIL.search(raw)
        sender_display = raw
        if email_m and len(raw) > len(email_m.group(0)) + 2:
            # es: "Ragione Sociale <mail@pec.it>"
            sender_display = raw
        elif email_m:
            sender_display = email_m.group(0)
        quote = re.sub(r"\s+", " ", (m.group(0))).strip()[:240]
        page = _page_of_match(text_ff, m.start())
        return sender_display, quote, page

    # 2.2 — fallback: cerca email “vicino” a indicatori PEC
    ctx_rx = re.compile(
        r"(messaggio\s+di\s+posta\s+certificata|postacert|daticert\.xml|ricevut[ae]\s+di\s+accettazione|avvenuta\s+consegna)",
        re.I
    )
    ctx = ctx_rx.search(text_ff)
    if ctx:
        start = max(0, ctx.start() - 400)
        end = min(len(text_ff), ctx.end() + 400)
        window = text_ff[start:end]
        em = _RX_EMAIL.search(window)
        if em:
            email = em.group(0)
            quote = re.sub(r"\s+", " ", window[max(0, em.start() - 40):em.end() + 40]).strip()[:240]
            page = _page_of_match(text_ff, start + em.start())
            return email, quote, page

    # 2.3 — ultimo tentativo: prima email nel documento
    em = _RX_EMAIL.search(text_ff)
    if em:
        email = em.group(0)
        quote = email
        page = _page_of_match(text_ff, em.start())
        return email, quote, page

    return None, None, None


def analyze_pec_document(text_ff: str) -> Dict:
    """
    Analizza il testo di UN documento come PEC e risponde alle 4 verifiche:
    1) Arrivata via PEC (art. 8 c.2)  -> presenza indicatori ricezione/accettazione
    2) Oggetto conforme               -> confronta con EXPECTED_SUBJECT (togli 'POSTA CERTIFICATA:')
    3) Nei termini temporali          -> data/ora compresa tra WINDOW_START e WINDOW_END
    4) Assunta a protocollo           -> presenza di n. protocollo (+ data protocollo)
    Ritorna un dict con chiavi e struttura: { "esito","dettaglio","evidenza":[{"quote","page"}] }
    """
    txt = text_ff or ""

    # --------------------- 1) PEC pervenuta ---------------------
    rx_receipts = [
        re.compile(r"messaggio\s+di\s+posta\s+certificata", re.I),
        re.compile(r"daticert\.xml", re.I),
        re.compile(r"ricevut[ae]\s+di\s+accettazione|avvenuta\s+consegna|ricevuta\s+di\s+consegna", re.I),
    ]
    m_receipt = None
    for rx in rx_receipts:
        m_receipt = rx.search(txt)
        if m_receipt:
            break
    is_pec = m_receipt is not None
    ev1_q = (re.sub(r"\s+", " ", m_receipt.group(0)).strip()[:240] if m_receipt else None)
    ev1_p = _page_of_match(text_ff, m_receipt.start()) if m_receipt else None

    def pack(ok: bool, detail: str, quote: Optional[str], page: Optional[int]) -> Dict:
        return {
            "esito": "Sì" if ok else "No",
            "dettaglio": detail,
            "evidenza": [] if not quote else [{"quote": quote, "page": page}],
        }

    # --------------------- 2) Oggetto conforme ------------------
    def _norm_subject(s: str) -> str:
        s = re.sub(r"^\s*posta\s+certificata\s*:\s*", "", s, flags=re.I)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return re.sub(r"\s+", " ", s).strip().lower()

    m_subj = re.search(r"(?:^|\n)\s*(?:oggetto|subject)\s*:\s*(.+)", txt, flags=re.I)
    subj_raw = m_subj.group(1).strip() if m_subj else ""
    subj_norm = _norm_subject(subj_raw) if subj_raw else ""
    exp_norm = _norm_subject(EXPECTED_SUBJECT)
    ok_subject = bool(subj_norm) and (exp_norm in subj_norm or subj_norm in exp_norm)
    subj_clean = re.sub(r"^\s*posta\s+certificata\s*:\s*", "", subj_raw, flags=re.I).strip()
    ev2_q = subj_raw[:240] if subj_raw else None
    ev2_p = _page_of_match(text_ff, m_subj.start()) if m_subj else None

    # --------------------- 3) Termini (finestra temporale) ------
    dt = _first_dt_in_text(txt)
    ok_window = bool(dt and (WINDOW_START <= dt <= WINDOW_END))
    ev3_q = None
    ev3_p = None
    if dt:
        for rx in _RX_DATES:
            mm = rx.search(txt)
            if mm:
                ev3_q = re.sub(r"\s+", " ", mm.group(0))[:240]
                ev3_p = _page_of_match(text_ff, mm.start())
                break

    # --------------------- 3.bis) Mittente PEC -----------------
    sender, evs_q, evs_p = _find_sender(txt)
    ok_sender = bool(sender)

    # --------------------- 4) Protocollo (numero + data) -------
    # Normalizza SOLO per parsing protocollo: togli spazi attorno a "/" e tra cifre
    txt_compact = re.sub(r"\s*/\s*", "/", txt)
    txt_compact = re.sub(r"(?<=\d)\s+(?=\d)", "", txt_compact)

    # Esempi coperti:
    # "Protocollo di ingresso N. 0267858 del 14/06/2024"
    # "Prot. n 1234/AB in data 01.07.2024"
    rx_proto_full = re.compile(
        r"(?:protocollo|prot\.?)\s*(?:di\s*ingresso|ingresso)?\s*(?:n\.?|num\.?)\s*"
        r"(?P<num>[A-Z0-9\/\.\-]{3,})"
        r"(?:\s*(?:del|in\s*data)\s*(?P<data>[0-3]?\d[\/\.-][01]?\d[\/\.-]\d{4}))?",
        re.I | re.S
    )
    m_full = rx_proto_full.search(txt_compact)
    proto_num = (m_full.group("num").strip() if m_full and m_full.group("num") else "")
    proto_date = (m_full.group("data").strip() if m_full and m_full.group("data") else "")
    ok_proto = bool(proto_num)

    # Evidenza (quote) dal testo originale, per mostrare riga leggibile
    rx_proto_line = re.compile(
        r"(?:protocollo|prot\.?).{0,80}(?:ingresso)?.{0,80}(?:n\.?|num\.?).{0,80}"
        r"(?:del|in\s*data)?.{0,20}\d{1,2}\s*[\/\.-]\s*\d{1,2}\s*[\/\.-]\s*\d{4}",
        re.I | re.S
    )
    mp_line = rx_proto_line.search(txt)
    if not mp_line:
        rx_proto_any = re.compile(r"(?:protocollo|prot\.?).{0,80}(?:n\.?|num\.?).{0,40}[A-Z0-9\/\.\-]{3,}", re.I | re.S)
        mp_line = rx_proto_any.search(txt)

    ev4_q = (re.sub(r"\s+", " ", mp_line.group(0)).strip()[:240]) if mp_line else (proto_num or None)
    ev4_p = _page_of_match(text_ff, mp_line.start()) if mp_line else None

    if ok_proto and proto_date:
        proto_detail = f"Protocollo: {proto_num} — Data: {proto_date}"
    elif ok_proto:
        proto_detail = f"Protocollo: {proto_num}"
    else:
        proto_detail = "Numero di protocollo non trovato."

    # --------------------- Ritorno risultati --------------------
    return {
        "pec_ai_sensi_art8": pack(
            is_pec,
            "Ricevuta/indicatori PEC rilevati." if is_pec else "Mancano riferimenti PEC/receipts.",
            ev1_q, ev1_p
        ),
        "oggetto_conforme": pack(
            ok_subject,
            (subj_raw if subj_raw else "Oggetto non trovato."),
            ev2_q, ev2_p
        ),
        "nei_termini": pack(
            ok_window,
            f"Data rilevata: {dt.strftime('%d/%m/%Y %H:%M:%S')}" if dt else "Data non trovata.",
            ev3_q, ev3_p
        ),
        "mittente_pec": pack(
            ok_sender,
            f"Mittente: {sender}" if sender else "Mittente non trovato.",
            evs_q, evs_p
        ),
        "protocollo": pack(
            ok_proto,
            proto_detail,
            ev4_q, ev4_p
        ),
        # campi raw comodi per export/uso downstream
        "protocollo_numero": proto_num,
        "protocollo_data": proto_date,
        "oggetto_valore": subj_clean,
        "mittente_valore": re.sub(r"^\s*Per conto di:\s*", "", sender or "", flags=re.I),
        "nei_termini_data": dt.strftime("%d/%m/%Y %H:%M:%S") if dt else "",
    }


# --------------------- RAG: embeddings, retriever, ask ---------------------
@st.cache_resource(show_spinner=False)
def get_embeddings() -> Embeddings:
    return VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=PROJECT_ID,
        location="europe-west4",
        credentials=GOOGLE_CREDENTIALS,
        client_options=ClientOptions(api_endpoint="europe-west4-aiplatform.googleapis.com"),
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
        credentials=GOOGLE_CREDENTIALS,
        temperature=0.2,
        max_output_tokens=8192,
    )


@st.cache_resource(show_spinner=False)
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " "],
    )


class LinesParser(BaseOutputParser):
    def parse(self, text: str) -> List[str]:
        return [line.strip() for line in text.split("\n") if line.strip()]


def docs_from_pdf_texts(text_by_file: Dict[str, str]) -> List[Document]:
    """Un Document per pagina, con metadata 'source', 'page' e 'preview'."""
    docs = []
    for filename, content in text_by_file.items():
        pages = content.split("\f")
        for p_idx, p_text in enumerate(pages, start=1):
            if p_text.strip():
                doc = Document(page_content=p_text, metadata={"source": filename, "page": p_idx})
                docs.append(doc)
    return docs


def build_retriever(text_by_file: Dict[str, str]):
    splitter = get_text_splitter()
    page_docs = docs_from_pdf_texts(text_by_file)
    chunks = splitter.split_documents(page_docs)

    # mini preview
    for d in chunks:
        preview = d.page_content[:300]
        d.metadata["preview"] = (preview + "…") if len(d.page_content) > 300 else preview

    vectordb = FAISS.from_documents(chunks, get_embeddings())
    sem_ret = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 50, "fetch_k": 300, "lambda_mult": 0.5}
    )
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 40

    base_retriever = EnsembleRetriever(retrievers=[bm25, sem_ret], weights=[0.45, 0.55])

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "DISTILLA la domanda seguente in una BASE QUERY corta per il retrieval, "
            "rimuovendo istruzioni/condizioni/soglie/formattazioni. Mantieni solo l'oggetto "
            "informativo principale (entità/attributo/valore) e i termini più discriminanti. "
            "Poi genera DIECI varianti robuste partendo da quella BASE QUERY."
            "Regole:"
            "- SOLO linee di testo, una per riga."
            "- Riga 1 = BASE QUERY."
            "- Righe 2–11 = varianti."
            "- Ogni linea ≤ 8–10 parole, senza punteggiatura inutile."
            "- Usa sinonimi, sigle, forme abbreviate, cifre e parole (es. '24' e 'ventiquattro')."
            "- Includi varianti con errori OCR plausibili (accenti/apostrofi, spaziature, 'identita' per 'identità')."
            "- Se compaiono confronti/limiti (≤, ≥, min/max), NON ripeterli nelle query: concentra la ricerca sul fatto da trovare "
            "(es. 'durata progetto', 'data invio PEC')."
            "- Se riguarda documenti personali, includi anche: documento di riconoscimento, carta identità, CI, c.i., "
            "documento identita, copia documento, fronte retro."
            "- Se riguarda PEC, includi: posta elettronica certificata, ricevuta di accettazione, ricevuta di consegna, "
            "avvenuta consegna, oggetto PEC."
            "- Se riguarda allegati nominali (B/C/C1/C2/C3/E), includi sia 'Allegato X' sia descrizioni equivalenti.\n\n"
            "Domanda: {question}"
        ),
    )

    query_chain: RunnableSequence = query_prompt | get_llm() | LinesParser()
    mqr = MultiQueryRetriever(
        retriever=base_retriever,
        llm_chain=query_chain,
        parser_key="lines"
    )
    return mqr, chunks


def _embed_docs(texts: List[str]) -> np.ndarray:
    embs = get_embeddings().embed_documents(texts)
    return np.array(embs, dtype="float32")


def _embed_query(q: str) -> np.ndarray:
    return np.array(get_embeddings().embed_query(q), dtype="float32")


IT_STOP = {
    "la", "il", "lo", "le", "i", "gli", "un", "una", "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "che", "come", "del", "della", "dello", "dei", "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle",
    "e", "ed", "o", "oppure", "non", "sono", "sia", "sul", "sulla", "sulle", "sugli", "nel", "nella", "nelle", "negli",
    "ai sensi", "art", "articolo", "dpr", "ue", "avviso", "oggetto", "mittente", "destinatario"
}


def extract_keywords(q: str) -> List[str]:
    qn = unicodedata.normalize("NFKD", q).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[A-Za-z0-9@._-]{3,}", qn.lower())
    toks = [t for t in tokens if t not in IT_STOP and len(t) > 2]
    cnt = Counter(toks)
    return [t for t, _ in cnt.most_common(6)]


def rerank_by_cosine_with_boost(
        query: str,
        docs: List[Document],
        liked_tags: Optional[Set[str]] = None,
        tags_by_source: Optional[Dict[str, Set[str]]] = None,
        top_n: int = 14,
        alpha: float = 0.84,
        beta: float = 0.10,
        gamma: float = 0.06,
) -> List[Document]:
    """
    Reranking soft:
      - alpha * cos_sim(query, chunk)
      - beta  * tag_boost (nome+contenuto)
      - gamma * keyword_boost (match parole chiave query nel chunk)
    Nessun filtro duro: i tag aiutano, non limitano.
    """
    if not docs:
        return docs

    # cosine similarity
    qv = _embed_query(query)
    m = _embed_docs([d.page_content for d in docs])
    sims = (m @ qv) / (np.linalg.norm(m, axis=1) * np.linalg.norm(qv) + 1e-9)

    # tag boost
    boosts_tag = np.zeros(len(docs), dtype="float32")
    if liked_tags and tags_by_source:
        for i, d in enumerate(docs):
            src = d.metadata.get("source")
            if not src:
                continue
            src_tags = tags_by_source.get(src, set())
            overlap = len(liked_tags & src_tags)
            if overlap > 0:
                boosts_tag[i] = min(1.0, 0.7 + 0.3 * overlap)

    # keyword boost
    kws = extract_keywords(query)
    boosts_kw = np.zeros(len(docs), dtype="float32")
    if kws:
        kw_re = re.compile("|".join(re.escape(k) for k in kws), re.I)
        for i, d in enumerate(docs):
            hits = len(kw_re.findall(d.page_content))
            if hits > 0:
                boosts_kw[i] = min(1.0, 0.2 + 0.15 * np.log1p(hits))

    scores = alpha * sims + beta * boosts_tag + gamma * boosts_kw
    order = np.argsort(-scores)
    return [docs[i] for i in order[:min(top_n, len(docs))]]


def _strip_header_from_quote(quote: str, src: str) -> str:
    pattern = rf'^\s*\[?\s*{re.escape(src)}\s*–\s*p\.\s*\d+\s*\]?\s*'
    return re.sub(pattern, "", quote or "", flags=re.IGNORECASE).strip()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _norm_quote_for_compare(s: str) -> str:
    # normalizza spazi e apostrofi semplici/tipografici
    s = s.replace("’", "'").replace("`", "'")
    return _norm_ws(s)


# ---------------------- Allegato B: parser SOLO TESTO ----------------------

def _ab_norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _ab_clean_value(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("…", " ").replace("·", " ").replace("—", " ").replace("–", " ")
    s = re.sub(r"[\.]{3,}", " ", s)  # "....." -> " "
    s = re.sub(r"(\.\s*){3,}", " ", s)  # ". . . . ." -> " "
    s = s.replace(" :", ":").replace(":", ": ")
    s = _ab_norm_space(s)
    s = re.sub(r"\.*$", "", s).strip()
    return s


# Lookahead per "tagliare" i valori fino alla prossima etichetta nota
_AB_NEXT_LABELS = [
    r"Denominazione\s*:", r"Natura giuridica\s*:", r"Posta elettronica certificata",
    r"Comune di\s*:", r"Prov\.\s*:", r"CAP\b", r"Via e n\. civ\.\s*:", r"Tel\.\s*:", r"Stato\s*:",
    # RL + varianti
    r"Cognome\s*:", r"Nome\s*:", r"Data di nascita\s*:",
    r"Provincia di nascita\s*:", r"Provincia\s*:",  # <— aggiunto
    r"Comune \(o Stato estero\) di nascita\s*:", r"C\.F\. firmatario\s*:", r"in qualità di",
    # Referente + varianti telefono
    r"Società\s*:\s*", r"\bCF\b", r"E-mail\s*:", r"Cellulare\s*:",  # <— aggiunto Cellulare
    # Progetto / final
    r"REFERENTE DA CONTATTARE", r"avente\s+per\s+titolo", r"Il\s+costo\s+complessivo",
    r"Luogo\s+e\s+data", r"FIRMA"
]
_AB_NEXT = r"(?=(?:%s))" % "|".join(_AB_NEXT_LABELS)
# --- Address & phone helpers ---
_ADDR_FROM_LABEL = re.compile(
    r"Via\s*(?:e\s*n\.?\s*civ\.?)?\s*:\s*(?P<addr>.+?)(?=\s+(?:Tel\.|Telefono|Stato\b|CAP\b)|$)",
    re.IGNORECASE
)
_ADDR_FALLBACK = re.compile(
    r"(?:Via|Viale|V\.le|Piazza|P\.zza|P\.le|Corso|C\.so|Largo|Strada|Contrada|C\.da|Localit[aà]|Loc\.)"
    r"\s+[A-Za-zÀ-ÖØ-öø-ÿ'’\. ]+?(?:\s*(?:n\.?|n°|num\.?|,)?\s*\d+\w?|(?:\s*s\.?n\.?c\.?|\s*snc))",
    re.IGNORECASE
)
_CLEAN_TEL = re.compile(r"\s*(?:Cellulare|Cell\.)\s*:\s*.*$", re.IGNORECASE)


def _ab_grab_after(label_regex: str, text: str, max_len: int = 300) -> str:
    r"""Prende il testo dopo l'etichetta fino alla prossima etichetta."""
    m = re.search(label_regex + r"(?P<val>.+)", text, flags=re.IGNORECASE)
    if not m:
        return ""
    val = m.group("val")
    m2 = re.search(_AB_NEXT, val, flags=re.IGNORECASE)
    if m2:
        val = val[:m2.start()]
    return _ab_clean_value(val[:max_len])


def parse_allegato_b_from_text(text_ff: str, source_filename: str = "Allegato_B.pdf") -> dict:
    """
    text_ff: testo dell'Allegato B (anche con \f tra pagine va bene).
    Ritorna un dizionario con tutti i campi estratti.
    """
    lines = [_ab_norm_space(l) for l in text_ff.splitlines() if _ab_norm_space(l)]
    joined = "\n".join(lines)
    big = " ".join(lines)

    # Ricava i confini della sezione 1 (SOGGETTO PROPONENTE) per evitare di
    # pescare CF del firmatario nella sezione 3.
    m_s2 = re.search(r"\n\s*2\.\s*SEDE\s+LEGALE", joined, flags=re.IGNORECASE)
    section1 = joined[:m_s2.start()] if m_s2 else joined

    out = {
        # 1. SOGGETTO PROPONENTE
        "denominazione": "",
        "codice_fiscale": "",
        "partita_iva": "",
        "natura_giuridica": "",
        "pec_proponente": "",
        # 2. SEDE LEGALE
        "sede_legale_comune": "",
        "sede_legale_provincia": "",
        "sede_legale_cap": "",
        "sede_legale_via_civico": "",
        "sede_legale_telefono": "",
        "sede_legale_stato": "",
        # 3. RAPPRESENTANTE LEGALE
        "rl_cognome": "",
        "rl_nome": "",
        "rl_data_nascita": "",
        "rl_provincia_nascita": "",
        "rl_comune_o_stato_estero_nascita": "",
        "rl_codice_fiscale": "",
        "rl_in_qualita_di": "",
        # 4. REFERENTE
        "ref_cognome": "",
        "ref_nome": "",
        "ref_societa": "",
        "ref_codice_fiscale": "",
        "ref_telefono": "",
        "ref_cellulare": "",
        "ref_email": "",
        # 5. PROGETTO
        "progetto_titolo": "",
        "progetto_durata_mesi": "",
        "progetto_costo_euro": "",
        # FINAL
        "file_origine": source_filename,
        "estrazione_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # ---------- (1) SOGGETTO PROPONENTE ----------
    out["denominazione"] = _ab_clean_value(_ab_grab_after(r"Denominazione\s*:\s*", section1))
    out["natura_giuridica"] = _ab_clean_value(_ab_grab_after(r"Natura giuridica\s*:\s*", section1))
    out["pec_proponente"] = _ab_clean_value(_ab_grab_after(r"Posta elettronica certificata.*?:\s*", section1))

    # CF + P.IVA sulla stessa riga (accetta 'P. IVA' / 'PIVA' / 'Partita IVA')
    m_cfpi = re.search(
        r"(?:C\.F\.|Codice\s*fiscale)\s*:\s*(?P<cf>[^P\n\r]+?)\s+"
        r"(?:P\.?\s*IVA|Partita\s+IVA)\s*[: ]*(?P<piva>[0-9A-Za-z\.\s]+)",
        section1, flags=re.IGNORECASE)
    if m_cfpi:
        out["codice_fiscale"] = _ab_clean_value(m_cfpi.group("cf"))
        out["partita_iva"] = _ab_clean_value(m_cfpi.group("piva"))
    else:
        # fallback separati nella sola sezione 1
        out["codice_fiscale"] = _ab_clean_value(_ab_grab_after(r"(?:C\.F\.|Codice\s*fiscale)\s*:\s*", section1))
        out["partita_iva"] = _ab_clean_value(_ab_grab_after(r"(?:P\.?\s*IVA|Partita\s+IVA)\s*[: ]*", section1))

    # 2) SEDE LEGALE (campi singoli → niente refusi)
    out["sede_legale_comune"] = _ab_clean_value(_ab_grab_after(r"Comune di\s*:\s*", joined))
    out["sede_legale_provincia"] = _ab_clean_value(_ab_grab_after(r"Prov\.\s*:\s*", joined))
    m_cap = re.search(r"CAP\s*[: ]*\s*(?P<cap>[0-9A-Za-z\.\- ]+)", joined, flags=re.IGNORECASE)
    out["sede_legale_cap"] = _ab_clean_value(m_cap.group("cap")) if m_cap else _ab_clean_value(
        _ab_grab_after(r"CAP\s*[: ]*", joined))

    # Isola la sezione 2 per catturare via/telefono in modo robusto
    m_s2blk = re.search(r"\n\s*2\.\s*SEDE\s+LEGALE.*?(?=\n\s*3\.)", joined, flags=re.IGNORECASE | re.S)
    sec2 = m_s2blk.group(0) if m_s2blk else joined

    # Via e n. civico (3 tentativi ordinati)
    addr = _ab_grab_after(r"Via e n\. civ\.\s*:\s*", sec2)
    if addr:
        # taglia eventuale "Tel./Telefono/Stato/CAP" sulla stessa riga
        addr = re.split(r"\s+(?:Tel\.|Telefono|Stato\b|CAP\b)\s*:? ", addr)[0].strip()
    else:
        m_label = _ADDR_FROM_LABEL.search(sec2)
        if m_label:
            addr = m_label.group("addr").strip()
        else:
            m_fb = _ADDR_FALLBACK.search(sec2)
            addr = m_fb.group(0).strip() if m_fb else ""
    out["sede_legale_via_civico"] = _ab_clean_value(addr)

    # Telefono (pulizia di eventuale "Cellulare: ...")
    tel = _ab_grab_after(r"Tel\.\s*:\s*", sec2) or _ab_grab_after(r"Telefono\s*:\s*", sec2)
    tel = _CLEAN_TEL.sub("", tel).strip()
    out["sede_legale_telefono"] = _ab_clean_value(tel)

    out["sede_legale_stato"] = _ab_clean_value(_ab_grab_after(r"Stato\s*:\s*", joined))

    # ---------- (3) RAPPRESENTANTE LEGALE ----------
    out["rl_cognome"] = _ab_clean_value(_ab_grab_after(r"(?m)^\s*Cognome\s*:\s*", joined))
    out["rl_nome"] = _ab_clean_value(_ab_grab_after(r"(?m)^\s*Nome\s*:\s*", joined))

    # Riga combinata: "Data di nascita: 15/08/1956 Provincia: CAGLIARI ..."
    m_dob = re.search(
        r"Data\s*di\s*nascita\s*:\s*(?P<data>[^\n\r]+?)"
        r"(?:\s+(?:Prov(?:incia)?(?:\s*di\s*nascita)?|Provincia)\s*:\s*(?P<prov>[^\n\r]+?))?"
        r"(?:\s+(?:Comune\s*\(o\s*Stato\s*estero\)\s*di\s*nascita|Comune\s*di\s*nascita)\s*:\s*(?P<comune>[^\n\r]+?))?"
        r"(?=$|\n|\r|Cognome\s*:|Nome\s*:|C\.F\. firmatario\s*:|in qualità di|E-mail\s*:|Tel\.\s*:|Cellulare\s*:)",
        joined, flags=re.IGNORECASE)
    if m_dob:
        out["rl_data_nascita"] = _ab_clean_value(m_dob.group("data"))
        if not out["rl_provincia_nascita"] and m_dob.group("prov"):
            out["rl_provincia_nascita"] = _ab_clean_value(m_dob.group("prov"))
        # NON forziamo 'comune' qui (lo estraiamo anche con la label specifica)
    else:
        out["rl_data_nascita"] = _ab_clean_value(_ab_grab_after(r"Data di nascita\s*:\s*", joined))

    # Province (anche se non c'è "di nascita")
    if not out["rl_provincia_nascita"]:
        out["rl_provincia_nascita"] = _ab_clean_value(_ab_grab_after(r"Provincia di nascita\s*:\s*", joined))
    if not out["rl_provincia_nascita"]:
        out["rl_provincia_nascita"] = _ab_clean_value(_ab_grab_after(r"Provincia\s*:\s*", joined))  # variante breve

    out["rl_comune_o_stato_estero_nascita"] = _ab_clean_value(
        _ab_grab_after(r"Comune \(o Stato estero\) di nascita\s*:\s*", joined)
    )
    out["rl_codice_fiscale"] = _ab_clean_value(_ab_grab_after(r"C\.F\. firmatario\s*:\s*", joined))
    out["rl_in_qualita_di"] = _ab_clean_value(_ab_grab_after(r"in qualità di\s*", joined))

    # ---------- (4) REFERENTE ----------
    if re.search(r"4\.\s*REFERENTE DA CONTATTARE", joined, flags=re.IGNORECASE):
        ref_section = joined.split("4. REFERENTE DA CONTATTARE", 1)[-1]
        out["ref_cognome"] = _ab_clean_value(_ab_grab_after(r"(?m)^\s*Cognome\s*:\s*", ref_section))
        out["ref_nome"] = _ab_clean_value(_ab_grab_after(r"(?m)^\s*Nome\s*:\s*", ref_section))
        out["ref_societa"] = _ab_clean_value(_ab_grab_after(r"Società\s*:\s*", ref_section))
        out["ref_codice_fiscale"] = _ab_clean_value(_ab_grab_after(r"\bCF\b\s*[: ]*", ref_section))
        out["ref_email"] = _ab_clean_value(_ab_grab_after(r"E-mail\s*:\s*", ref_section))
        out["ref_telefono"] = _ab_clean_value(_ab_grab_after(r"Tel\.\s*:\s*", ref_section))
        out["ref_cellulare"] = _ab_clean_value(_ab_grab_after(r"Cellulare\s*:\s*", ref_section))
    else:
        out["ref_societa"] = _ab_clean_value(_ab_grab_after(r"Società\s*:\s*", joined))
        out["ref_codice_fiscale"] = _ab_clean_value(_ab_grab_after(r"\bCF\b\s*[: ]*", joined))
        out["ref_email"] = _ab_clean_value(_ab_grab_after(r"E-mail\s*:\s*", joined))

    # ---------- (5) PROGETTO ----------
    m = re.search(
        r"avente\s+per\s+titolo\s+(?P<titolo>.+?)\s+della\s+prevista\s+durata\s+di\s+n\s*(?P<mesi>[0-9]+|\w+)\s*mesi",
        big, flags=re.IGNORECASE)
    if m:
        out["progetto_titolo"] = _ab_clean_value(m.group("titolo"))
        out["progetto_durata_mesi"] = _ab_clean_value(m.group("mesi"))
    else:
        out["progetto_titolo"] = _ab_clean_value(_ab_grab_after(r"avente\s+per\s+titolo\s+", big))
        m3 = re.search(r"durata\s+di\s+n\s*(?P<mesi>[0-9]+|\w+)\s*mesi", big, flags=re.IGNORECASE)
        if m3:
            out["progetto_durata_mesi"] = _ab_clean_value(m3.group("mesi"))

    m_cost = re.search(r"Il\s+costo\s+complessivo\s+previsto\s+è\s+di\s+euro\s*(?P<costo>[0-9\.\, ]+)", big,
                       flags=re.IGNORECASE)
    out["progetto_costo_euro"] = _ab_clean_value(m_cost.group("costo")) if m_cost else _ab_clean_value(
        _ab_grab_after(r"costo\s+complessivo.*?euro\s*", big))

    # Fallback business rule: impresa individuale ⇒ CF proponente = CF RL (se mancante)
    if not out["codice_fiscale"] and out.get("rl_codice_fiscale") and "IMPRESA INDIVIDUALE" in out.get(
            "natura_giuridica", "").upper():
        out["codice_fiscale"] = out["rl_codice_fiscale"]

    return out


def make_downloads_for_ab(data: dict, base_name: str = "allegato_b"):
    """Restituisce (json_bytes, csv_bytes, csv_sep)."""
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        import pandas as pd  # usa pandas se disponibile
        import io
        df = pd.DataFrame([data])
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep=";")
        csv_bytes = buf.getvalue().encode("utf-8-sig")
        sep = ";"
    except Exception:
        # csv minimale senza pandas
        import io
        keys = list(data.keys())
        row = ";".join([str(data.get(k, "")).replace(";", ",") for k in keys])
        buf = io.StringIO()
        buf.write(";".join(keys) + "\n")
        buf.write(row + "\n")
        csv_bytes = buf.getvalue().encode("utf-8-sig")
        sep = ";"
    return json_bytes, csv_bytes, sep


def _page_text_for(docs: List[Document], src: str, page) -> str:
    for d in docs:
        if d.metadata.get("source") == src and str(d.metadata.get("page")) == str(page):
            return d.page_content
    return ""


def _auto_extract_evidence(question: str, docs: List[Document]) -> Optional[Dict]:
    """Se la LLM non produce una quote valida, prova a trovarne una vicino alle keyword della query."""
    kws = extract_keywords(question)
    if not kws:
        return None
    kw_re = re.compile("|".join(re.escape(k) for k in kws), re.I)
    for d in docs[:6]:
        txt = d.page_content
        m = kw_re.search(txt)
        if m:
            start = max(0, m.start() - 160)
            end = min(len(txt), m.end() + 160)
            quote = re.sub(r"\s+", " ", txt[start:end]).strip()[:240]
            if quote:
                return {"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", "?"), "quote": quote}
    return None


def ask_rag(
        question: str,
        retriever,
        llm,
        liked_tags: Optional[Set[str]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Ritorna (final_answer, evidence_list).
    Modalità RAG-only: nessun filtro. Tag/keyword solo come boost in reranking.
    Ha un fallback che tenta un'estrazione automatica di una quote se la LLM non ne produce una valida.
    """
    docs = retriever.invoke(question)
    docs = rerank_by_cosine_with_boost(
        question, docs, liked_tags=liked_tags, tags_by_source=st.session_state.doc_tags, top_n=14
    )
    if not docs:
        return "No – non presente nei documenti.", []

    # CONTEXT in blocchi <doc ...>
    context_blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        pg = d.metadata.get("page", "?")
        txt = d.page_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        context_blocks.append(f'<doc source="{src}" page="{pg}">\n{txt}\n</doc>')
    context = "\n".join(context_blocks)

    # Prompt con richieste di JSON + una sola evidenza
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sei un assistente legale. Rispondi SOLO con informazioni presenti nei documenti forniti."
                "\n- Inizia SEMPRE con 'Sì' o 'No', e DOPO aggiungi in UNA SOLA FRASE il dettaglio chiave utile (es. indirizzo PEC mittente, oggetto, data/ora invio, nominativo su documento, ecc.)."
                "\n- Se l'informazione non è nel contesto, rispondi: 'No – non presente nei documenti.'"
                "\n- Il CONTEXT è in blocchi <doc source=\"...\" page=\"...\"> ... </doc>."
                "\n- Le QUOTE devono essere estratte VERBATIM dal testo interno ai tag <doc>."
                "\n- NON considerare come evidenza una frase che dice solo di allegare o richiedere un documento."
                "\n- Restituisci SOLO questo JSON:"
                '\n{\n  "final_answer": "<risposta breve in italiano>",'
                '\n  "evidence": [{"source":"<file.pdf>","page":<numero>,"quote":"<estratto max 240 caratteri>"}]'
                "\n}"
                "\n- Restituisci AL MASSIMO 1 evidenza."
            ),
            (
                "human",
                "QUESTION:\n{{ question }}\n\nCONTEXT:\n{{ context }}\n\n"
                "Regole:\n"
                "- Massimo 1 evidenza.\n"
                "- 'quote' deve essere testuale dal CONTEXT.\n"
                "- Se non trovi evidenze, metti evidence: [].\n"
                "- SOLO il JSON, senza testo extra.\n"
            ),
        ],
        template_format="jinja2",
    )

    msgs = prompt.format_messages(question=question, context=context)
    raw = llm.invoke(msgs).content

    def _safe_parse_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r'\{.*\}', s, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    data = _safe_parse_json(raw)
    if not isinstance(data, dict) or "final_answer" not in data or "evidence" not in data:
        final_answer = raw.strip()[:1200] if raw else "No – non presente nei documenti"
        return final_answer, []

    final_answer = str(data.get("final_answer", "")).strip()
    ev_list = data.get("evidence", []) or []

    # Validazione minima della quote contro il testo
    validated: List[Dict] = []
    for ev in ev_list[:1]:
        try:
            src = str(ev.get("source", "?")).strip()
            pg = ev.get("page", "?")
            try:
                pg = int(pg)
            except Exception:
                pg = str(pg)
            q = _strip_header_from_quote(str(ev.get("quote", ""))[:240], src)
            page_txt = _page_text_for(docs, src, pg)
            if q and page_txt:
                qt = _norm_quote_for_compare(q)
                pt = _norm_quote_for_compare(page_txt)
                if qt and qt in pt:
                    validated.append({"source": src, "page": pg, "quote": q})
        except Exception:
            continue

    # Fallback: se risposta è Sì ma manca una quote valida, prova auto-estrazione vicino alle keyword
    if final_answer.lower().startswith("sì") and not validated:
        auto_ev = _auto_extract_evidence(question, docs)
        if auto_ev:
            validated = [auto_ev]

    # Se ancora senza evidenza, degradare
    if final_answer.lower().startswith("sì") and not validated:
        return "No – evidenza non trovata nei testi.", []

    return final_answer, validated

def ask_rag_detailed(
    question: str,
    retriever,
    llm,
    liked_tags: Optional[Set[str]] = None,
    max_docs: int = 20,
    max_evidences: int = 5
) -> str:
    docs = retriever.invoke(question)
    docs = rerank_by_cosine_with_boost(
        question, docs,
        liked_tags=liked_tags,
        tags_by_source=st.session_state.doc_tags,
        top_n=max_docs
    )

    # CONTEXT in blocchi <doc>
    context_blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        pg  = d.metadata.get("page", "?")
        txt = d.page_content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        context_blocks.append(f'<doc source="{src}" page="{pg}">\n{txt}\n</doc>')
    context = "\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Sei un assistente legale. Rispondi SOLO usando il CONTEXT fornito (testo dentro i tag <doc …> … </doc>). "
             "Non aggiungere conoscenze esterne né interpretazioni creative.\n\n"
             "FORMATTAZIONE OBBLIGATORIA (in questo esatto ordine):\n"
             "## Analisi per punti\n"
             "- Produci da 3 a 10 bullet sintetici e operativi. Ogni bullet deve riflettere informazioni ESPLICITE nel CONTEXT. "
             "Se qualcosa non è presente nel CONTEXT, aggiungi un bullet tipo: 'Informazione mancante: …'.\n\n"
             "## Valutazione a/b/c\n"
             "- Per ciascuna lettera scrivi SOLO 'Sì', 'No' oppure 'Incertezza', seguito da una riga di motivazione. "
             "Se la domanda non definisce i tre aspetti, scegli tu tre criteri pertinenti e indicali tra parentesi (es. '(Presenza)') "
             "prima del giudizio. Niente paragrafi lunghi.\n\n"
             "## Evidenze\n"
             "- Inserisci fino a {{max_evidences}} citazioni VERBATIM, ognuna come blocco quote Markdown (riga che inizia con '>'), "
             "lunghezza massima 240 caratteri; subito sotto specifica in corsivo il riferimento *(file – p.X)*.\n"
             "- La prima citazione deve essere la più decisiva e, nelle domande di attribuzione/responsabilità (chi paga/sostiene/è responsabile/intestatario), "
             "deve contenere nella stessa frase o in adiacenza immediata sia l’azione sia il soggetto.\n"
             "- Se non trovi citazioni adeguate, scrivi: 'Nessuna evidenza adeguata trovata nel CONTEXT.' e non concludere 'Sì'.\n\n"
             "REGOLE:\n"
             "1) Usa solo frasi presenti nel CONTEXT per trarre conclusioni. Se i testi sono ambigui o contraddittori, indica la contraddizione e scegli 'Incertezza'.\n"
             "2) Le citazioni sono VERBATIM (nessuna parafrasi). Fuori dalle citazioni, resta sintetico. "
             "3) Non riportare intere pagine; resta nei limiti. 4) Rispondi in italiano."
             ),
            ("human",
             "DOMANDA:\n{{ question }}\n\nCONTEXT:\n{{ context }}\n\n"
             "Applica rigorosamente il formato e le regole sopra.")
        ],
        template_format="jinja2",
    )

    msgs = prompt.format_messages(
        question=question,
        context=context,
        max_evidences=max_evidences
    )
    return llm.invoke(msgs).content  # Markdown completo

def _make_context_blocks_for_bullets(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "?")
        pg  = d.metadata.get("page", "?")
        txt = d.page_content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        blocks.append(f'<doc source="{src}" page="{pg}">\n{txt}\n</doc>')
    return "\n".join(blocks)

def _validate_bullet_evidence(ev_list: List[Dict], docs: List[Document], max_items: int = 3) -> List[Dict]:
    """Tiene solo le evidenze che compaiono davvero nel testo del chunk indicato (max 3)."""
    out = []

    def page_text(src, pg):
        for d in docs:
            if d.metadata.get("source")==src and str(d.metadata.get("page"))==str(pg):
                return d.page_content
        return ""

    for ev in (ev_list or [])[:max_items]:
        try:
            src = str(ev.get("source","?")).strip()
            pg  = ev.get("page","?")
            try: pg = int(pg)
            except: pg = str(pg)
            q   = _strip_header_from_quote(str(ev.get("quote",""))[:240], src)
            txt = page_text(src, pg)
            if q and txt and (_norm_quote_for_compare(q) in _norm_quote_for_compare(txt)):
                out.append({"source":src,"page":pg,"quote":q})
        except Exception:
            continue
    return out

def _pick_docs_for_bullets(question: str, retriever, liked_tags=None, top_n: int = 18) -> List[Document]:
    docs = retriever.invoke(question)
    return rerank_by_cosine_with_boost(
        question, docs, liked_tags=liked_tags,
        tags_by_source=st.session_state.doc_tags, top_n=top_n
    )

def ask_rag_bullets(
    question: str,
    retriever,
    llm,
    liked_tags: Optional[Set[str]] = None,
    max_docs: int = 18,
    min_bullets: int = 3,
    max_bullets: int = 6
) -> Dict:
    """
    Ritorna un payload:
    {
      "verdict": "Sì|No|Incertezza",
      "bullets": ["...", "..."],
      "evidence": [{"source":"file.pdf","page":N,"quote":"..."}]
    }
    """
    docs = _pick_docs_for_bullets(question, retriever, liked_tags=liked_tags, top_n=max_docs)
    if not docs:
        return {"verdict":"Incertezza","bullets":["Nessun documento pertinente recuperato."],"evidence":[]}

    context = _make_context_blocks_for_bullets(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Sei un assistente legale. Usa SOLO il CONTEXT per rispondere alla DOMANDA.\n"
             "Restituisci SEMPRE e SOLO un JSON con le chiavi esatte:\n"
             '{ "verdict":"Sì|No|Incertezza", "bullets":[...], '
             '"evidence":[{"source":"<file>","page":<n>,"quote":"<≤240 caratteri, VERBATIM>"}] }\n'
             "Regole di decisione (universali):\n"
             f"- bullets: da {min_bullets} a {max_bullets} punti, frasi brevi e operative, "
             "ancorate a ciò che c’è (o manca) nel CONTEXT.\n"
             "- evidence: 1–3 citazioni VERBATIM dal CONTEXT; la PRIMA deve essere la più decisiva.\n"
             "- 'Sì' SOLO se almeno una citazione prova direttamente il nucleo della domanda.\n"
             "  • Per domande di ATTRIBUZIONE (chi paga/chi è responsabile/intestatario…): "
             "la citazione deve contenere nella stessa frase (o adiacenza immediata) il predicato "
             "('a carico di', 'sostenute da', 'responsabile', 'intestatario'…) e l’attore "
             "(beneficiari/proponente/ente X). In mancanza → non rispondere 'Sì'.\n"
             "  • Per NUMERI/DATE/SOGLIE: la citazione deve riportare il valore esplicito.\n"
             "  • Per PRESENZA/CONFORMITÀ: serve una dicitura testuale esplicita (titolo/intestazione/regola).\n"
             "- 'No' quando il CONTEXT contiene una frase che contraddice la condizione richiesta (citala).\n"
             "- Se i testi sono insufficienti/ambigui/contraddittori → 'Incertezza'.\n"
             "- Se non trovi citazioni adeguate, non usare 'Sì' e imposta 'evidence': [].\n"
             "- Citazioni senza parafrasi, ≤240 caratteri. Nessun testo fuori dal JSON. Rispondi in italiano."
             ),
            ("human",
             "DOMANDA:\n{{ question }}\n\nCONTEXT:\n{{ context }}\n\n"
             "Rispondi SOLO con il JSON richiesto.")
        ],
        template_format="jinja2",
    )

    msgs = prompt.format_messages(question=question, context=context)
    raw = llm.invoke(msgs).content

    # parsing robusto
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, flags=re.S)
        if m:
            try: data = json.loads(m.group(0))
            except Exception: data = None

    if not isinstance(data, dict):
        data = {"verdict":"Incertezza","bullets":["Risposta non in JSON valido."],"evidence":[]}

    # normalizza campi
    data.setdefault("verdict", "Incertezza")
    data.setdefault("bullets", [])
    data.setdefault("evidence", [])

    # valida evidenze
    data["evidence"] = _validate_bullet_evidence(data.get("evidence"), docs, max_items=3)

    # se "Sì" ma senza evidenze valide, degrada a Incertezza
    if str(data["verdict"]).strip().lower().startswith("sì") and not data["evidence"]:
        data["verdict"] = "Incertezza"
        if len(data["bullets"]) < max_bullets:
            data["bullets"].append("Evidenze non verificabili nei testi: servono riferimenti precisi.")

    # limita bullets
    if not isinstance(data["bullets"], list):
        data["bullets"] = [str(data["bullets"])]
    data["bullets"] = [b for b in data["bullets"] if b][:max_bullets]
    while len(data["bullets"]) < min_bullets:
        data["bullets"].append("Informazione parziale o non conclusiva nel CONTEXT.")

    return data

def render_bullets_result(payload: Dict, section_key: str):
    verdict = payload.get("verdict","Incertezza")
    bullets = payload.get("bullets", [])
    evs     = payload.get("evidence", [])

    icon = "✅" if verdict=="Sì" else ("❌" if verdict=="No" else "⚠️")
    st.markdown(f"**Esito:** {icon} {verdict}")

    if bullets:
        st.markdown("**Perché:**")
        for b in bullets:
            st.markdown(f"- {b}")

    if evs:
        st.markdown("**Evidenze:**")
        for i, e in enumerate(evs, start=1):
            st.markdown(f"- _{e['source']} – p.{e['page']}_")
            st.markdown(f"> {e['quote']}")
            toggle_key = stable_key('blt_prev_open', section_key, i, e['source'], e['page'])
            open_preview = st.checkbox("👁️ Mostra/chiudi anteprima", key=toggle_key)
            if open_preview:
                show_pdf_page_image(e["source"], e["page"])



# --------------------- Pipeline file (estrazione e tag) ---------------------
def process_uploaded_file(file) -> Optional[str]:
    """Estrae testo, fa OCR se serve, assegna TAG da nome e da contenuto (solo boost), salva bytes."""
    name = file.name
    ext = Path(name).suffix.lower()
    bytes_data = file.read()

    # Estrazione PDF
    if ext == ".pdf":
        pdf_bytes = bytes_data
    elif ext == ".p7m":
        pdf_bytes = extract_pdf_from_p7m(bytes_data)
        st.session_state.signers[name] = estrai_firmatari(bytes_data)
        if not pdf_bytes:
            st.warning(f"{name}: errore nell'estrazione da P7M.")
            return None
    else:
        st.warning(f"{name}: formato non supportato.")
        return None

    # salva pdf per anteprima
    st.session_state.pdf_blobs[name] = pdf_bytes

    # Estrai testo grezzo
    text = extract_pdf_text(pdf_bytes)
    sample = " ".join(text.split()[:200])
    if not is_readable(sample):
        st.info(f"{name}: testo non leggibile, eseguo OCR…")
        text = run_ocr_with_document_ai(pdf_bytes)

    # TAG (solo per boost/UX)
    tags_fn = classify_by_filename(name)
    tags_tx = classify_by_content(text)

    st.session_state.tag_by_filename[name] = tags_fn
    st.session_state.tag_by_text[name] = tags_tx

    # --- Precedenza nome file e singolarità Allegato_* ---
    allegati_fn = {t for t in tags_fn if t.startswith("Allegato_")}
    allegati_tx = {t for t in tags_tx if t.startswith("Allegato_")}

    primary = None
    if allegati_fn:
        for t in ALLEGATO_TAG_ORDER:
            if t in allegati_fn:
                primary = t
                break
    elif allegati_tx:
        for t in ALLEGATO_TAG_ORDER:
            if t in allegati_tx:
                primary = t
                break

    # Se c'è un Allegato_* primario, rimuovi gli altri per evitare rumore
    if primary:
        for t in list(allegati_fn | allegati_tx):
            if t != primary:
                tags_fn.discard(t)
                tags_tx.discard(t)

    all_tags = tags_fn.union(tags_tx)

    st.session_state.texts[name] = text
    st.session_state.doc_tags[name] = all_tags
    for t in all_tags:
        st.session_state.tag_index.setdefault(t, set()).add(name)

    st.success(f"{name}: indicizzato. Tag (boost): {', '.join(sorted(all_tags)) or '—'}")
    return text


# ------------------------------ UI helpers ---------------------------------
def show_pdf_page_image(src_name: str, page: int, zoom: float = 1.6):
    """Renderizza la pagina del PDF come immagine e la mostra."""
    pdf_bytes = st.session_state.get("pdf_blobs", {}).get(src_name)
    if not pdf_bytes:
        st.warning("PDF non disponibile in memoria per l’anteprima.")
        return
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            idx = max(0, int(page) - 1)
            if idx >= doc.page_count:
                st.warning(f"Pagina {page} fuori intervallo (pagine totali: {doc.page_count}).")
                return
            mat = fitz.Matrix(zoom, zoom)
            pix = doc[idx].get_pixmap(matrix=mat, alpha=False)
            st.image(pix.tobytes("png"), caption=f"{src_name} – pagina {page}", use_container_width=True)
    except Exception as e:
        st.error(f"Errore anteprima pagina: {e}")


def render_evidence_block(evidence: List[Dict], section_id: str, title: str = "📎 Evidenza"):
    if not evidence:
        return
    e = evidence[0]  # una sola evidenza
    src, pg, quote = e["source"], e["page"], e["quote"]
    st.markdown(f"**{title}**  \n_{src} – p.{pg}_")
    st.markdown(f"> {quote}")

    toggle_key = stable_key("prev_open_free", section_id, src, pg)
    open_preview = st.checkbox("👁️ Mostra/chiudi anteprima", key=toggle_key)
    if open_preview:
        show_pdf_page_image(src, pg)


# ---------------------- Allegati (regex forti, no RAG) ---------------------
def _compile_rx(p: str) -> re.Pattern:
    return re.compile(p, re.I | re.S | re.M)


# Le chiavi sono in ordine logico: PRESENZA subito sopra la corrispondente FIRMA

ATT_RULES: Dict[str, Dict] = {
    # -------------------- Allegato B --------------------
    "Allegato B – Presenza": {
        "any_of": [
            _compile_rx(r"allegato\s*b\b.{0,600}\bdomanda\s+di\s+partecipazione\b"),
        ],
        "prefer_tags": {"Allegato_B"},
    },

    # -------------------- Allegato C --------------------
    "Allegato C – Presenza": {
        "any_of": [
            _compile_rx(
                r"allegato\s*c(?!\d)\b.{0,600}(?:\bdichiarazione\s+soggetto\s+proponente\b|dpr\s*445/2000|art\.?\s*46)"
            ),
        ],
        "prefer_tags": {"Allegato_C"},
    },

    # -------------------- Allegato C1 -------------------
    "Allegato C1 – Presenza": {
        "any_of": [
            _compile_rx(r"allegato\s*c1\b.{0,600}\b(?:descrizion[ea]\s+progetto|titolo\s+progetto)\b"),
        ],
        "prefer_tags": {"Allegato_C1"},
    },

    # -------------------- Allegato C2 -------------------
    "Allegato C2 – Presenza": {
        "any_of": [
            _compile_rx(
                r"allegato\s*c2\b.{0,600}(?:(?:detag?lio|quadro)\s+finanziari\w*(?:\s+del\s+progetto)?|attivit[aà]\s+finanziabil[ei])"
            ),
        ],
        "prefer_tags": {"Allegato_C2"},
    },

    # -------------------- Allegato C3 -------------------
    "Allegato C3 – Presenza": {
        "any_of": [
            _compile_rx(
                r"allegato\s*c3\b.{0,600}\b(?:ripartizion[ea]\s+territoriale\s+(?:del\s+progetto|degli\s+interventi)|regolamento\s*\(?\s*ue\s*\)?\s*2022/2472)"
            ),
        ],
        "prefer_tags": {"Allegato_C3"},
    },

    "Allegato D – Presenza": {
        "any_of": [
            _compile_rx(r"allegato\s*d\b.{0,400}\bdichiarazion[ei]\s+soggetto\s+beneficiario\b"),
            _compile_rx(r"\bdichiarazion[ei]\s+soggetto\s+beneficiario\b.{0,400}art\.?\s*46\s+dpr\s*445/2000"),
        ],
        "prefer_tags": {"Allegato_D"},
    },

    # -------------------- Allegato E --------------------
    "Allegato E – Presenza": {
        "any_of": [
            _compile_rx(
                r"allegato\s*e\b.{0,600}\bdichiarazione\s+di\s+impegno\s+alla\s+costituzione\s+della\s+filiera\s+biologic[aa]\b"
            ),
        ],
        "prefer_tags": {"Allegato_E"},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Tagging robusto Allegati: riuso regex "forti" (le stesse di ATT_RULES).
# Considera SOLO l'intestazione (prima pagina) per evitare match su semplici citazioni.
# ─────────────────────────────────────────────────────────────────────────────
ALLEGATO_TAG_ORDER = ["Allegato_B", "Allegato_C", "Allegato_C1", "Allegato_C2", "Allegato_C3", "Allegato_D",
                      "Allegato_E"]

# Mappa tag -> patterns "any_of" riusando ATT_RULES (coerenza con la checklist)
TAG_ANY_OF = {
    "Allegato_B": ATT_RULES["Allegato B – Presenza"]["any_of"],
    "Allegato_C": ATT_RULES["Allegato C – Presenza"]["any_of"],
    "Allegato_C1": ATT_RULES["Allegato C1 – Presenza"]["any_of"],
    "Allegato_C2": ATT_RULES["Allegato C2 – Presenza"]["any_of"],
    "Allegato_C3": ATT_RULES["Allegato C3 – Presenza"]["any_of"],
    "Allegato_D": ATT_RULES["Allegato D – Presenza"]["any_of"],
    "Allegato_E": ATT_RULES["Allegato E – Presenza"]["any_of"],
}


def _head(text_ff: str, max_chars: int = 6000) -> str:
    """Ritorna la prima pagina (o i primi max_chars) per evitare match su semplici citazioni."""
    first_page = text_ff.split("\f", 1)[0]
    return first_page[:max_chars]


def detect_attachment_tags(text_ff: str) -> Set[str]:
    """Applica le regex 'forti' alla sola intestazione per decidere il tag Allegato_*."""
    head = _head(text_ff)
    out: Set[str] = set()
    for tag, patterns in TAG_ANY_OF.items():
        if _match_any_of(head, patterns):
            out.add(tag)
    # se matchano più tag, scegli un primario coerente
    for t in ALLEGATO_TAG_ORDER:
        if t in out:
            return {t}
    return out


def _page_of_match(text_ff: str, match_start: int) -> int:
    """Calcola pagina (1-based) dato l'indice di inizio match su testo con form feed."""
    pages = text_ff.split("\f")
    off = 0
    for i, p in enumerate(pages, start=1):
        nxt = off + len(p)
        if match_start <= nxt:
            return i
        off = nxt + 1
    return max(1, len(pages))


def _find_one(text: str, rx: re.Pattern) -> Optional[Tuple[str, int]]:
    m = rx.search(text)
    if not m:
        return None
    quote = re.sub(r"\s+", " ", m.group(0)).strip()[:240]
    page = _page_of_match(text, m.start())
    return (quote, page)


def _match_all_of(text: str, patterns: List[re.Pattern]) -> Optional[List[Tuple[str, int]]]:
    out: List[Tuple[str, int]] = []
    for rx in patterns:
        found = _find_one(text, rx)
        if not found:
            return None
        out.append(found)
    return out


def _match_any_of(text: str, patterns: List[re.Pattern]) -> Optional[Tuple[str, int]]:
    for rx in patterns:
        found = _find_one(text, rx)
        if found:
            return found
    return None


def check_attachment_rule(rule_label: str, texts_by_file: Dict[str, str]) -> Tuple[str, List[Dict]]:
    """
    Applica la regola indicata:
      - 'all_of': tutte le pattern devono comparire nello stesso file
      - 'any_of': almeno una pattern deve comparire nello stesso file
      - 'prefer_tags': ordina i file preferendo quelli con quei tag
      - 'restrict_to_tags': limita la ricerca ai soli file con quei tag
    Ritorna: (risposta breve, evidenza max 1).
    """
    rule = ATT_RULES.get(rule_label)
    if not rule:
        return "No – regola allegato non definita.", []

    all_of: List[re.Pattern] = rule.get("all_of", [])
    any_of: List[re.Pattern] = rule.get("any_of", [])
    prefer_tags: Set[str] = set(rule.get("prefer_tags", set()))
    restrict_to_tags: Set[str] = set(rule.get("restrict_to_tags", set()))

    files = list(texts_by_file.keys())

    # restrizione per tag (es. le regole 'Firma' sul relativo allegato)
    if restrict_to_tags:
        files = [f for f in files if st.session_state.doc_tags.get(f, set()) & restrict_to_tags]

    # ordinamento: prima i file con i tag preferiti
    if prefer_tags:
        preferred = [f for f in files if st.session_state.doc_tags.get(f, set()) & prefer_tags]
        others = [f for f in files if f not in preferred]
        files = preferred + others

    # scorri i file finché una combinazione soddisfa la regola
    for fname in files:
        text = texts_by_file[fname]

        # ALL OF
        if all_of:
            ev_all = _match_all_of(text, all_of)
            if not ev_all:
                continue

        # ANY OF
        if any_of:
            ev_any = _match_any_of(text, any_of)
            if not ev_any:
                continue

        # costruzione evidenza (max 1 snippettino)
        evidence: List[Dict] = []
        if all_of:
            q, p = (_match_all_of(text, all_of) or [("", 1)])[0]
            evidence.append({"source": fname, "page": p, "quote": q})
        elif any_of:
            q, p = ev_any
            evidence.append({"source": fname, "page": p, "quote": q})

        return "Sì – trovato nel documento selezionato.", evidence

    return "No – elementi richiesti non trovati nello stesso documento.", []


# ---------------------- Domande preimpostate (solo RAG) --------------------
@st.cache_resource(show_spinner=False)
def load_preset_questions(path: str = "domande_preimpostate.json") -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Filtra via le domande sulla “presenza allegati”: ora gestite da regex
            SKIP = {
                "Presenza Allegato B",
                "Firma digitale Allegato B",
                "Presenza e firma Allegato C",
                "Presenza e firma Allegato C1",
                "Presenza e firma Allegato C2",
                "Presenza e firma Allegato C3",
                "Presenza e firma Allegato E",
                "Presenza Allegato B – (se c'era variante)",
            }
            return {k: v for k, v in data.items() if k not in SKIP}
    except Exception as e:
        st.error(f"Errore nel caricamento delle domande preimpostate: {e}")
        return {}

@st.cache_resource(show_spinner=False)
def load_questions(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Errore nel caricamento domande '{path}': {e}")
        return {}


# --------------------------------- UI --------------------------------------
st.title("📄 Analisi Ricevibilità - Ammissibilità")

# Session state init
if "texts" not in st.session_state:
    st.session_state.texts: Dict[str, str] = {}
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "preset_answers" not in st.session_state:
    st.session_state.preset_answers: Dict[str, str] = {}
if "check_results" not in st.session_state:
    st.session_state.check_results = {}
if "signers" not in st.session_state:
    st.session_state.signers = {}
if "last_free_evidence" not in st.session_state:
    st.session_state.last_free_evidence = []
if "check_sources" not in st.session_state:
    st.session_state.check_sources = {}
if "pdf_blobs" not in st.session_state:
    st.session_state.pdf_blobs = {}
if "doc_tags" not in st.session_state:
    st.session_state.doc_tags: Dict[str, Set[str]] = {}
if "tag_index" not in st.session_state:
    st.session_state.tag_index: Dict[str, Set[str]] = {}
if "tag_by_filename" not in st.session_state:
    st.session_state.tag_by_filename: Dict[str, Set[str]] = {}
if "tag_by_text" not in st.session_state:
    st.session_state.tag_by_text: Dict[str, Set[str]] = {}
if "ab_extract_result" not in st.session_state:
    st.session_state.ab_extract_result = None

# Sidebar – Upload e pipeline
with st.sidebar:
    st.header("📂 Carica i file")
    files = st.file_uploader(
        "Carica uno o più PDF o P7M",
        type=["pdf", "p7m"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # KEY STABILE
    if st.button("📥 Estrai testo", key="btn_extract"):
        # reset completo
        st.session_state.texts = {}
        st.session_state.doc_tags = {}
        st.session_state.tag_index = {}
        st.session_state.tag_by_filename = {}
        st.session_state.tag_by_text = {}
        st.session_state.signers = {}
        st.session_state.pdf_blobs = {}
        st.session_state.retriever = None
        st.session_state.llm = None
        st.session_state.last_free_evidence = []
        st.session_state.check_results = {}
        st.session_state.check_sources = {}

        if not files:
            st.warning("Nessun file caricato.")
        else:
            for file in files:
                process_uploaded_file(file)

        if st.session_state.texts:
            st.success("✅ Testi estratti e indicizzati. Tag (solo boost) assegnati.")

    if st.session_state.texts:
        # 🔼 Sposto il bottone QUI, sopra alla selezione documenti
        if st.button("▶️ Analizza ricevibilità", key="btn_sidebar_pec_top"):
            with st.spinner("Analizzo la ricevibilità per ciascun documento…"):
                st.session_state.pec_results = {
                    fname: analyze_pec_document(text)
                    for fname, text in st.session_state.texts.items()
                }
            st.session_state.show_pec_results = True
            st.success("Analisi PEC completata. Scorri la pagina per vedere i risultati.")

        # --- poi la selezione dei documenti per la RAG ---
        st.markdown("### 🔎 Seleziona i documenti per l'addestramento")
        selected = []
        for idx, nome in enumerate(st.session_state.texts):
            sel = st.checkbox(f"{nome}", value=True, key=make_key("sel", idx, nome))
            if sel:
                selected.append(nome)

        # --- e infine il bottone di addestramento ---
        if st.button("📚 Addestra Sistema - Ammissibilità", key="btn_build_rag"):
            sel_texts = {n: st.session_state.texts[n] for n in selected} if selected else st.session_state.texts
            retriever, _ = build_retriever(sel_texts)
            st.session_state.retriever = retriever
            st.session_state.llm = get_llm()
            st.success(f"📚 Addestramento terminato! Documenti indicizzati: {len(sel_texts)}")

# Main – Lista file + tag
with st.expander("📋 File caricati e tag assegnati", expanded=False):
    for nome in st.session_state.texts:
        t_fn = sorted(st.session_state.tag_by_filename.get(nome, []))
        t_tx = sorted(st.session_state.tag_by_text.get(nome, []))
        all_tags = sorted(set(t_fn) | set(t_tx))
        icona = "🔐" if st.session_state.signers.get(nome) else "📄"
        st.markdown(f"- {icona} `{nome}` — **Tag (boost):** {', '.join(all_tags) if all_tags else '—'}")
        if all_tags:
            st.caption(f"Da nome: {', '.join(t_fn) or '—'} · Da contenuto: {', '.join(t_tx) or '—'}")

# Anteprima testi estratti
with st.expander("📝 Anteprima testi estratti", expanded=False):
    for idx, (nome, testo) in enumerate(st.session_state.texts.items()):
        with st.expander(f"📄 {nome}", expanded=False):
            anteprima = " ".join(testo.split()[:500])
            st.text_area(
                "Testo estratto",
                anteprima,
                height=250,
                disabled=True,
                key=make_key("preview_text", idx, nome, len(testo)),
            )

# ---------------------- Firme digitali (mostra solo dopo estrazione) ----------------------
if st.session_state.texts:  # se ci sono testi estratti, vuol dire che hai già caricato/elaborato i file
    st.subheader("🔐 Firme digitali rilevate")
    trovati = False
    for nome, firmatari in st.session_state.signers.items():
        if not firmatari:
            continue
        trovati = True
        with st.expander(f"{nome}"):
            for i, info in enumerate(firmatari, 1):
                if "Errore" in info:
                    st.error(f"Errore durante la lettura della firma: {info['Errore']}")
                else:
                    st.markdown(f"**Firmatario #{i}**")
                    st.write(f"- **CN**: {info['CN']}")
                    st.write(f"- **Emesso da**: {info['Emesso da']}")
                    st.write(f"- **Valido dal**: {info['Valido dal']}")
                    st.write(f"- **Valido fino**: {info['Valido fino']}")
                    st.divider()

    if not trovati:
        st.info("Nessuna firma digitale rilevata nei file caricati (.p7m).")

        # ----- Risultati ricevibilità PEC (render se disponibili) -----

        # ----- Risultati ricevibilità PEC (render se disponibili) -----
    if st.session_state.get("pec_results"):
        st.markdown("---")
        st.header("📬 Analisi della ricevibilità (PEC) – risultati")
        import pandas as pd

        rows = []

        for fname, res in st.session_state.pec_results.items():
            st.subheader(f"📄 {fname}")


            def _badge(v: dict) -> str:
                ok = str(v.get("esito", "")).lower().startswith("sì")
                return "✅ Sì" if ok else "❌ No"


            # --- valori "puliti" come nel CSV ---
            oggetto_val = res.get("oggetto_valore", "") or "—"
            nei_term_data = res.get("nei_termini_data", "") or "—"
            mittente_val = res.get("mittente_valore", "") or "—"
            proto_num = res.get("protocollo_numero", "") or "—"
            proto_dt = res.get("protocollo_data", "") or ""

            # Righe riassuntive coerenti al CSV (niente prefissi nel valore)
            st.markdown(
                f"- **Domanda pervenuta tramite PEC (art. 8 c.2)**: {_badge(res['pec_ai_sensi_art8'])}"
            )
            st.markdown(
                f"- **Oggetto conforme (art. 8 c.2)**: {_badge(res['oggetto_conforme'])} — _{oggetto_val}_"
            )
            st.markdown(
                f"- **Presentata nei termini (DM 243212/2024)**: {_badge(res['nei_termini'])} — {nei_term_data}"
            )
            st.markdown(
                f"- **Mittente PEC**: {_badge(res['mittente_pec'])} — {mittente_val}"
            )
            st.markdown(
                f"- **Assunta al protocollo**: {_badge(res['protocollo'])} — {proto_num}"
                + (f" — {proto_dt}" if proto_dt else "")
            )


            # Evidenze con anteprima pagina
            def _render_ev(label: str, payload: dict):
                ev = payload.get("evidenza") or []
                if not ev:
                    return
                e = ev[0]
                st.caption(f"📎 Evidenza {label}: p.{e.get('page', '?')} — “{e.get('quote', '')[:200]}”")
                if st.button("👁️ Anteprima pagina", key=stable_key("pec_prev", fname, label, e.get('page', '?'))):
                    show_pdf_page_image(fname, e.get('page', 1))


            # Raccogli la prima evidenza utile tra le 4
            # Evidenza unica + anteprima apri/chiudi (expander)
            # Ordine di priorità: mittente -> oggetto -> termini -> protocollo -> ricevibilità
            priority = ["mittente_pec", "oggetto_conforme", "nei_termini", "protocollo", "pec_ai_sensi_art8"]

            first_ev = None
            first_key = None
            for k in priority:
                evs = res.get(k, {}).get("evidenza") or []
                if evs:
                    first_ev = evs[0]
                    first_key = k
                    break

            if first_ev:
                page = first_ev.get("page", "?")
                quote = first_ev.get("quote", "").strip().replace("\n", " ")
                quote_short = (quote[:200] + "…") if len(quote) > 200 else quote

                st.caption(f"📎 Evidenza ({first_key}): p.{page} — “{quote_short}”")
                with st.expander(f"👁️ Anteprima documento — {fname}", expanded=False):
                    show_pdf_page_image(fname, page)

            st.divider()

            rows.append({
                "file": fname,
                "PEC (art.8 c.2)": res["pec_ai_sensi_art8"]["esito"],

                "Oggetto conforme": res["oggetto_conforme"]["esito"],
                "Oggetto": res.get("oggetto_valore", ""),

                "Nei termini": res["nei_termini"]["esito"],
                "Data rilevata": res.get("nei_termini_data", ""),

                "Protocollo": res["protocollo"]["esito"],
                "Numero Protocollo": res.get("protocollo_numero", ""),
                "Data Protocollo": res.get("protocollo_data", ""),

                "Mittente PEC": res["mittente_pec"]["esito"],
                "Mittente": res.get("mittente_valore", ""),
            })

        df = pd.DataFrame(rows)
        st.markdown("### 📦 Esporta risultati")
        st.dataframe(df, use_container_width=True)

        # --- export: proteggi zeri iniziali in Excel ---
        df_export = df.copy()


        def _protect_excel_text(x: str) -> str:
            x = (str(x) if x is not None else "").strip()
            return f'="{x}"' if x else ""


        if "Numero Protocollo" in df_export.columns:
            df_export["Numero Protocollo"] = df_export["Numero Protocollo"].map(_protect_excel_text)

        csv_str = df_export.to_csv(index=False, sep=";")
        csv_bytes = ("\ufeff" + csv_str).encode("utf-8")  # BOM per Excel

        json_bytes = json.dumps(st.session_state.pec_results, ensure_ascii=False, indent=2).encode("utf-8")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Scarica CSV (;) risultati PEC",
                               data=csv_bytes, file_name="pec_ricevibilita.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Scarica JSON risultati PEC",
                               data=json_bytes, file_name="pec_ricevibilita.json",
                               mime="application/json", use_container_width=True)

# ---------------------------- Q&A + Check-list ------------------------------
if st.session_state.retriever and st.session_state.llm:
    st.markdown("---")

    # Domanda libera (UN SOLO INPUT, con key stabile)
    st.header("❓ Domanda libera sui documenti")
    query = st.text_input("Scrivi la tua domanda", key="free_query_input")


    def infer_liked_from_text(t: str) -> Set[str]:
        liked = set()
        tl = t.lower()
        if "pec" in tl:
            liked.add("PEC")
        if "allegato b" in tl:
            liked.add("Allegato_B")
        if "allegato c1" in tl:
            liked.add("Allegato_C1")
        if "allegato c2" in tl:
            liked.add("Allegato_C2")
        if "allegato c3" in tl:
            liked.add("Allegato_C3")
        if "allegato c" in tl and not {"Allegato_C1", "Allegato_C2", "Allegato_C3"} & liked:
            liked.add("Allegato_C")
        if "allegato d" in tl:
            liked.add("Allegato_D")
        if "allegato e" in tl:
            liked.add("Allegato_E")
        return liked


    if st.button("🔍 Rispondi", key="btn_free_answer") and query.strip():
        with st.spinner("Sto cercando nei documenti…"):
            answer, evidence = ask_rag(
                query,
                st.session_state.retriever,
                st.session_state.llm,
                liked_tags=infer_liked_from_text(query) or None
            )
        st.session_state.last_free_evidence = evidence or []
        st.markdown("### 📢 Risposta:")
        st.success(answer)

    # Mostra le varianti SOLO se c'è una query non vuota
    if SHOW_VARIANTS and query.strip() and isinstance(st.session_state.retriever, MultiQueryRetriever):
        st.markdown("### 🧠 Varianti generate:")
        try:
            variants = st.session_state.retriever.llm_chain.invoke({"question": query})
            for variant in variants:
                st.markdown(f"- {variant}")
        except Exception as e:
            st.error(f"Errore nel generare le varianti: {e}")

    if st.session_state.last_free_evidence:
        render_evidence_block(st.session_state.last_free_evidence, section_id="free")

        # ----------------- Checklist Allegati (regex forti, no RAG) -------------
    st.markdown("---")
    st.header("📌 Presenza allegati (Regex)")

    # CSS migliorato (compatibile con dark/light mode)
    st.markdown("""
    <style>
    .allegati-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    .allegato-card {
      border-radius: 10px;
      border: 1px solid #444;
      padding: 1rem 1.2rem;
      background-color: rgba(255,255,255,0.02); /* trasparente, adatta a dark/light */
    }
    .allegato-title {
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: .4rem;
    }
    .badge {
      display:inline-block;
      padding: .25rem .6rem;
      border-radius: .5rem;
      font-size:.82rem;
      font-weight:600;
    }
    .badge-ok {color:#0f5132; background:#d1e7dd; border:1px solid #badbcc;}
    .badge-ko {color:#842029; background:#f8d7da; border:1px solid #f5c2c7;}
    .badge-na {color:#999; background:#e0e0e0; border:1px solid #ccc;}
    .answer-text {
      font-size: .85rem;
      margin-top: .6rem;
      opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Stato risultati allegati
    if "att_results" not in st.session_state:
        st.session_state.att_results = {}
    if "att_ran_once" not in st.session_state:
        st.session_state.att_ran_once = False

    # Comandi
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("▶️ Verifica tutti", key="btn_att_check_all"):
            for label in ATT_RULES.keys():
                ans, ev = check_attachment_rule(label, st.session_state.texts)
                st.session_state.att_results[label] = {"ans": ans, "ev": ev}
            st.session_state.att_ran_once = True
    with colB:
        if st.button("♻️ Reset risultati", key="btn_att_reset"):
            st.session_state.att_results = {}
            st.session_state.att_ran_once = False

    # Griglia card con anteprima pagina/evidenza (apri/chiudi)
    st.markdown('<div class="allegati-grid">', unsafe_allow_html=True)

    for label in ATT_RULES.keys():
        res = st.session_state.att_results.get(label)

        ev = (res or {}).get("ev") or []
        ev0 = ev[0] if ev else None
        ev_src = ev0.get("source") if ev0 else None
        ev_page = ev0.get("page") if ev0 else None
        ev_quote = (ev0.get("quote", "") if ev0 else "").strip()

        if not res:
            badge_html = '<span class="badge badge-na">Non verificato</span>'
            icon = "❔"
            answer = "_Premi verifica per controllare_"
        else:
            ok = str(res["ans"]).lower().startswith("sì")
            badge_html = (
                '<span class="badge badge-ok">Presente</span>' if ok
                else '<span class="badge badge-ko">Assente</span>'
            )
            icon = "✅" if ok else "❌"
            answer = res["ans"]

        # Card HTML (titolo + esito)
        st.markdown(
            f"""
            <div class="allegato-card">
              <div class="allegato-title">{icon} {label}</div>
              <div>{badge_html}</div>
              <div class="answer-text">{answer}</div>
            """,
            unsafe_allow_html=True
        )

        # Se c'è un'evidenza, mostra quote + toggle anteprima
        if ev0:
            quote_short = (ev_quote[:200] + "…") if len(ev_quote) > 200 else ev_quote
            st.caption(f"📎 Evidenza: _{ev_src} – p.{ev_page}_")
            if quote_short:
                st.markdown(f"> {quote_short}")

            # Toggle persistente: apre/chiude l’anteprima pagina
            toggle_key = stable_key("att_prev_open", label, ev_src, ev_page)
            open_preview = st.checkbox("👁️ Anteprima pagina", key=toggle_key)

            if open_preview:
                show_pdf_page_image(ev_src, ev_page)

        # chiudi card
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Inline controls per Allegato B SE presente (stessa sezione "Presenza allegati")
    # ─────────────────────────────────────────────────────────────────────────────
    ab_res = st.session_state.att_results.get("Allegato B – Presenza")
    ab_ok = bool(ab_res and str(ab_res.get("ans", "")).lower().startswith("sì"))

    if ab_ok:
        # Trova il/i file candidato/i: 1) evidenza regex, 2) tag Allegato_B, 3) fallback per contenuto
        ev = (ab_res.get("ev") or [])
        from_ev = [ev[0]["source"]] if ev else []

        tagged = [f for f, t in st.session_state.doc_tags.items() if "Allegato_B" in t and f not in from_ev]

        rx1 = re.compile(r"\ballegato\s*b\b", re.I)
        rx2 = re.compile(r"domanda\s+di\s+partecipazione", re.I)
        by_text = [f for f, txt in st.session_state.texts.items() if rx1.search(txt) and rx2.search(txt)]
        by_text = [f for f in by_text if f not in from_ev and f not in tagged]

        candidates = from_ev + tagged + by_text
        candidates = [c for c in candidates if c in st.session_state.texts]  # safety

        st.markdown("#### ➡️ Estrai dati **Allegato B**")
        if not candidates:
            st.info("Nessun documento candidato trovato nonostante il match regex. Controlla i tag o il contenuto.")
        else:
            # selezione file (preseleziono quello dell’evidenza)
            default_idx = 0
            sel_file = st.selectbox(
                "Documento riconosciuto come Allegato B:",
                options=candidates,
                index=default_idx,
                key=stable_key("ab_inline_sel", "|".join(candidates))
            )

            colX, colY = st.columns([0.35, 0.65])
            with colX:
                if st.button("▶️ Estrai campi Allegato B", key=stable_key("btn_ab_inline", sel_file)):
                    txt = st.session_state.texts.get(sel_file, "")
                    data = parse_allegato_b_from_text(txt, source_filename=sel_file)
                    st.session_state.ab_extract_result = {"file": sel_file, "data": data}

            # mostra ultimo risultato (persistente tra rerun finché non si resetta)
            if st.session_state.ab_extract_result:
                res_file = st.session_state.ab_extract_result["file"]
                res_data = st.session_state.ab_extract_result["data"]

                st.success(f"Estrazione completata dal documento: **{res_file}**")
                with st.expander("📄 JSON estratto", expanded=True):
                    st.json(res_data)

                # download JSON/CSV
                jb, cb, sep = make_downloads_for_ab(res_data, base_name=os.path.splitext(os.path.basename(res_file))[0])
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇️ Scarica JSON",
                        data=jb,
                        file_name=f"{os.path.splitext(os.path.basename(res_file))[0]}_allegato_b.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with d2:
                    st.download_button(
                        f"⬇️ Scarica CSV (sep '{sep}')",
                        data=cb,
                        file_name=f"{os.path.splitext(os.path.basename(res_file))[0]}_allegato_b.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # ----------------- Check-list RAG (tutto il resto) ----------------------
    st.markdown("---")
    st.header("✅ Analisi ammissibilità")

    mode = st.radio(
        "Scegli tipo di analisi",
        ["Primo blocco (brevi)", "Secondo blocco (complesse)"],
        horizontal=True,
        key="amm_mode"
    )

    # Confronto robusto sul testo scelto
    if "brevi" in mode.lower():
        QUESTIONS = load_questions("domande_preimpostate_Semplici.json")
    else:
        QUESTIONS = load_questions("domande_preimpostate_Complesse.json")


    # util per chiavi stabili per-riga
    def stable_id(prefix: str, label: str) -> str:
        return f"{prefix}_{hashlib.md5(label.encode()).hexdigest()[:8]}"


    # Bar comandi
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        if st.button("🏃‍♂️ Esegui tutte le verifiche", key="run_all_mode"):
            for label, prompt in QUESTIONS.items():
                liked = infer_liked_from_text(label + " " + prompt)
                with st.spinner(f"Analisi: {label}…"):
                    payload = ask_rag_bullets(prompt, st.session_state.retriever, st.session_state.llm,
                                              liked_tags=liked or None, max_docs=20)
                    st.session_state.check_results[label] = payload
                    st.session_state.check_sources[label] = []  # non usato qui
            st.success("Verifiche completate.")
    with c2:
        if st.button("♻️ Reset", key="reset_mode"):
            st.session_state.check_results = {}
            st.session_state.check_sources = {}
            st.rerun()

    for label, prompt in QUESTIONS.items():
        res = st.session_state.check_results.get(label)
        col1, col2, col3 = st.columns([0.45, 0.40, 0.15])
        with col1:
            st.markdown(f"**{label}**")
            st.caption(prompt)
        with col3:
            if st.button("Esegui", key=f"run_{hash(label) % 10 ** 8}"):
                liked = infer_liked_from_text(label + " " + prompt)
                with st.spinner(f"Analisi: {label}…"):
                    payload = ask_rag_bullets(prompt, st.session_state.retriever, st.session_state.llm,
                                              liked_tags=liked or None, max_docs=20)
                    st.session_state.check_results[label] = payload
                    st.session_state.check_sources[label] = []
                st.rerun()

        with col2:
            res = st.session_state.check_results.get(label)
            if res:
                with st.expander("📄 Risposta dettagliata", expanded=True):
                    render_bullets_result(res, section_key=label)
        st.divider()
