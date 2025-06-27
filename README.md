# 🧠 Analizzatore Istruttorie AI

Questa app analizza documenti PDF e P7M, estrae il testo (anche tramite OCR se necessario), indicizza i contenuti e permette di verificare automaticamente il rispetto dei requisiti di un bando grazie a LLM (Gemini) e LangChain.

## 📁 Struttura

istruttorie-ai/
├── app/
│ └── rag_app_upload.py ← Codice principale Streamlit
├── .env.example ← Template per le chiavi API
├── requirements.txt ← Dipendenze Python
├── .gitignore ← Esclude file privati/locali
└── README.md ← Questo file

bash
Copia
Modifica

## ▶️ Come avviare l'app

### 1. Clona il repository

```bash
git clone https://github.com/tuo-utente/istruttorie-ai.git
cd istruttorie-ai
2. Crea un ambiente virtuale
bash
Copia
Modifica
python -m venv .venv
.\.venv\Scripts\activate  # Su Windows
3. Installa le dipendenze
bash
Copia
Modifica
pip install -r requirements.txt
4. Imposta le chiavi API
Copia il file .env.example in .env e compila le tue chiavi:

ini
Copia
Modifica
API_KEY=la_tua_api_key_di_gemini
DOC_AI_PROJECT_ID=nome_del_progetto
DOC_AI_LOCATION=eu
DOC_AI_PROCESSOR_ID=id_del_processor
❗ Il file .env NON deve essere caricato su GitHub.

5. Avvia l’app
bash
Copia
Modifica
streamlit run app/rag_app_upload.py
Una volta avviata, l'app sarà disponibile nel browser all'indirizzo http://localhost:8501 (o simile).

👥 Autore
BMTI ScpA – Ufficio AI
📧 Inserire contatto aziendale

yaml
Copia
Modifica

