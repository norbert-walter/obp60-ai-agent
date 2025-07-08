import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
import glob
import requests
from bs4 import BeautifulSoup

# --- OpenAI API-Key aus secrets.toml laden ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- UI: Streamlit Interface ---
st.set_page_config(page_title="Technischer Berater", page_icon="🛠️")
st.title("🛠️ Technischer Berater für Produktfragen")

# --- Mehrere PDFs aus einem Ordner laden ---
pdf_folder = "daten/"  # <-- Ersetze durch deinen Ordnerpfad
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

raw_text = ""
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        raw_text += page.extract_text()

# --- HTML-Webseiten einbinden ---
urls = [
    #"https://obp60-v2-docu.readthedocs.io/de/latest/index.html",
    #"https://obp40-v1-docu.readthedocs.io/de/latest/index.html",
    #"https://open-boat-projects.org"
    # Weitere URLs hier ergänzen
]

for url in urls:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        texts_from_html = []
        for element in soup.find_all(["p", "h1", "h2", "h3"]):
            texts_from_html.append(element.get_text())
        html_text = "\n".join(texts_from_html)
        raw_text += "\n" + html_text
    except Exception as e:
        st.warning(f"Fehler beim Laden der URL {url}: {e}")

# --- Text aufteilen ---
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

# --- Embeddings + Vektor-Datenbank erstellen ---
embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(texts, embeddings)
retriever = db.as_retriever()

# --- Conversation Chain mit Speicher ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    memory=memory
)

# --- Chat Interaktion ---
st.write("Stelle deine Frage zu unseren Produkten:")
user_question = st.text_input("Was möchtest du wissen?")
if user_question:
    response = qa_chain.run(user_question)
    st.write(response)
