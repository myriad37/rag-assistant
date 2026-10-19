import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def create_vector_store():
    data_path = "data"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print(f"Error: Please put some PDF documents inside the '{data_path}' folder.")
        return None

    print("Loading documents...")
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing Google Gemini embeddings...")
    # Use gemini-embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("Embedding chunks and storing in ChromaDB...")
    persist_directory = "./chroma_db"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Vector database successfully created and saved at '{persist_directory}'!")

if __name__ == "__main__":
    create_vector_store()