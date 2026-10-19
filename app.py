import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="Enterprise Document Q&A Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Enterprise Document Q&A Assistant")
st.write("Upload documents and chat with them securely using grounded AI.")

# --- Sidebar for Document Management ---
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

PERSIST_DIRECTORY = "./chroma_db"
DATA_DIRECTORY = "./data"

if uploaded_files:
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
    if st.sidebar.button("Process Uploaded Documents"):
        with st.spinner("Processing and embedding documents..."):
            # Load newly uploaded files
            all_docs = []
            for file in os.listdir(DATA_DIRECTORY):
                if file.endswith(".pdf"):
                    loader = PyMuPDFLoader(os.path.join(DATA_DIRECTORY, file))
                    all_docs.extend(loader.load())
            
            # Chunk
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_documents(all_docs)
            
            # Embed & Save
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            st.sidebar.success(f"Successfully processed {len(chunks)} chunks!")
            # Clear cache so the app reloads the new database
            st.cache_resource.clear()

# --- Load RAG Chain ---
@st.cache_resource
def get_rag_chain():
    if not os.path.exists(PERSIST_DIRECTORY):
        return None, "Please upload and process documents using the sidebar to begin."

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    template = """You are a precise corporate assistant. Answer the user's question using ONLY the provided context below. 
If you do not know the answer or if it's not explicitly written in the context, state "I cannot find the answer in the provided documents." 
Always cite the source document name and page number for the claims you make.

Context:
{context}

Question: {question}

Answer (with source citations):"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "Unknown Page")
            formatted.append(f"Source: {source} (Page {page})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, None

rag_chain, error_message = get_rag_chain()

if error_message:
    st.info(error_message)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")