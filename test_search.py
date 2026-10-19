from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def test_search():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Load the existing database from disk
    vector_store = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings
    )

    query = "What chemical reaction creates bioluminescence?"  # Change to fit your PDF

    print(f"\nRunning semantic search for query: '{query}'\n" + "-" * 40)

    results = vector_store.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source File: {doc.metadata.get('source')}")
        print(f"Page Number: {doc.metadata.get('page', 'Unknown')}")
        print(f"Snippet: {doc.page_content[:300]}...")

if __name__ == "__main__":
    test_search()