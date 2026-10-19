import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def setup_rag_chain():
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        print("Error: Chroma database not found! Please run Day 2 script first.")
        return None

    # 1. Load the existing Vector Database
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    # 2. Create a Retriever
    # We ask it to fetch the top 4 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 3. Initialize the LLM (Updated model name)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    # 4. Design the RAG Prompt Template
    template = """You are a precise corporate assistant. Answer the user's question using ONLY the provided context below. 
If you do not know the answer or if it's not explicitly written in the context, state "I cannot find the answer in the provided documents." 
Always cite the source document name and page number for the claims you make.

Context:
{context}

Question: {question}

Answer (with source citations):"""

    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format retrieved documents into a single text block
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "Unknown Page")
            formatted.append(f"Source: {source} (Page {page})\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # 5. Build the LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return vector_store, rag_chain


if __name__ == "__main__":
    vector_store, chain = setup_rag_chain()
    
    if chain:
        test_question = "What are the main takeaways or core financial highlights?"
        
        print(f"\nAsking Question: '{test_question}'\n" + "-" * 50)
        
        # Run the chain
        response = chain.invoke(test_question)
        print(response)
        
        # Print raw retrieved source files for verification
        print("\n" + "=" * 50 + "\nRetrieved Sources:")
        docs = vector_store.similarity_search(test_question, k=4)
        for doc in docs:
            print(f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")