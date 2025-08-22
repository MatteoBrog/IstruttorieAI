NON HO VOGLIA DI SCRIVERLO E LO FACCIO CON GPT. POI LO SISTEMO PROMESSO

# 🧠 Istruttorie AI

Applicazione **Streamlit** per l’analisi automatica di documenti PDF/P7M.  
L’app estrae i testi (con OCR tramite **Google Document AI** se necessario), li indicizza e permette di:

- ⚡ Eseguire verifiche automatiche sui requisiti di un bando
- 🔍 Porre domande libere sui documenti (RAG con **LangChain + Gemini**)
- 📑 Controllare la presenza e la firma degli allegati richiesti
- 🔐 Rilevare firme digitali nei file `.p7m`

---

## 📁 Struttura del progetto

IstruttorieAI/
│── app/
│ └── Demo_App.py # Codice principale Streamlit
│── domande_preimpostate.json # Domande standard per la check-list
│── requirements.txt # Dipendenze Python
│── README.md # Questo file
│── .gitignore # Esclude file sensibili/rigenerabili
│── .env.example # Esempio variabili locali
│── credentials.example.json # Esempio credenziali Google Service Account
│── secrets.example.toml # Esempio configurazione per Streamlit Cloud

yaml
Copia
Modifica

---

## ▶️ Avvio in locale

1. **Clona il repository**
   ```bash
   git clone https://github.com/tuo-utente/IstruttorieAI.git
   cd IstruttorieAI
Crea un ambiente virtuale

bash
Copia
Modifica
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac
Installa le dipendenze

bash
Copia
Modifica
pip install -r requirements.txt
Configura le credenziali

Copia .env.example → .env e inserisci i tuoi valori.

Copia credentials.example.json → credentials.json e incolla le credenziali del tuo Service Account GCP.
⚠️ Questo file non deve mai essere caricato su GitHub.

Per il deploy su Streamlit Cloud, usa secrets.example.toml come riferimento e incolla i dati reali in Project → Settings → Secrets.

Avvia l’app

bash
Copia
Modifica
streamlit run app/Demo_App.py
L’app sarà accessibile su http://localhost:8501.

🔑 Gestione credenziali
In locale

.env → contiene configurazioni base (ID progetto, location, ecc.)

credentials.json → chiavi del Service Account GCP

Su Streamlit Cloud

Copia il contenuto di secrets.example.toml (con valori reali) in
Project → Settings → Secrets

Lo script leggerà automaticamente st.secrets

📌 Note importanti
Non committare mai:

.env

credentials.json

.streamlit/secrets.toml

cartelle locali (.venv/, .idea/, .streamlit/, operatori/, ecc.)

Usa solo i file *.example come riferimento da includere nel repo.

👥 Autore
BMTI ScpA – Artificial Intelligence Hub
📧 [Inserire contatto aziendale]

yaml
Copia
Modifica

---

👉 Questa versione spiega:  
- cosa fa il progetto,  
- la struttura delle cartelle,  
- come avviarlo in locale,  
- come gestire i segreti in `.env`, `credentials.json`, `secrets.toml`.  

Vuoi che ti prepari anche il **requirements.txt definitivo** (basato sugli import che 