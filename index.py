from src.helper import load_pdf,split_text,download_hugging_face_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import time
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

pc = Pinecone()

index_name = "medical-chat-bot"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

docs = load_pdf('Data/')
doc_splits = split_text(docs)
embeddings = download_hugging_face_embedding()

docsearch = PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    index_name=index_name
)