import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

load_dotenv()


def load_documents(data_path="data"):
  """Loads all PDF and text documents from the specified directory."""
  if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(
        f"Created '{data_path}' directory. Please add some PDF files there and"
        " re-run."
    )
    return []

  # PyMuPDFLoader is fast and handles layout-heavy PDFs well
  loader = DirectoryLoader(
      data_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader
  )
  docs = loader.load()

  print(f"Successfully loaded {len(docs)} document pages.")
  return docs


if __name__ == "__main__":
  documents = load_documents()
  if documents:
    print(f"Sample preview from page 1:\n{documents[0].page_content[:300]}...")