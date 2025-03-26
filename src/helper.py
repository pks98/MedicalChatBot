# Import necessary modules from LangChain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def split_text(docs):
    """
    Splits documents into smaller chunks for better processing.

    Parameters:
    docs (list): List of document objects to be split.

    Returns:
    list: A list of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(documents=docs)
    return doc_splits

def load_pdf(dir_path):
    """
    Loads all PDF files from a given directory.

    Parameters:
    dir_path (str): Path to the directory containing PDF files.

    Returns:
    list: A list of loaded documents.
    """
    loader = DirectoryLoader(path=dir_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def download_hugging_face_embedding():
    """
    Downloads and initializes a pre-trained Hugging Face embedding model.

    Returns:
    HuggingFaceEmbeddings: An instance of the embedding model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)