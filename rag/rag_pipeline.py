from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = DATA_DIR / "vector" / "store"

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")


def load_all_text(folder_path: Path = DATA_DIR):
    all_docs = []
    folder_path.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(folder_path.glob("*.txt")):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {file_path.name}, total docs: {len(all_docs)}")

    return all_docs


def split_docs(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, local_files_only=True)

    def generate_embeddings(self, text):
        return self.model.encode(text, show_progress_bar=True)


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: Path = VECTOR_DIR,
        collection_name: str = "txt_documents",
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.collection = None
        self.client = None

    def initialize_store(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Text document retrieval store"},
        )

    def is_empty(self) -> bool:
        if self.collection is None:
            self.initialize_store()
        return self.collection.count() == 0

    def add_documents(self, embeddings, documents):
        if self.collection is None:
            self.initialize_store()

        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts do not match")

        ids = []
        metadatas = []
        document_texts = []
        embedding_rows = []

        for index, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc-{uuid.uuid4()}"
            metadata = dict(doc.metadata)
            metadata["doc_id"] = doc_id
            metadata["doc_index"] = index
            metadata["content_length"] = len(doc.page_content)
            ids.append(doc_id)
            metadatas.append(metadata)
            document_texts.append(doc.page_content)
            embedding_rows.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=document_texts,
            embeddings=embedding_rows,
        )


class RAGRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        if self.vector_store.collection is None:
            self.vector_store.initialize_store()

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_docs = []
        if results.get("documents") and results["documents"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for rank, (doc_id, metadata, document, dist) in enumerate(
                zip(ids, metadatas, docs, distances),
                start=1,
            ):
                similarity_score = 1 - dist
                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": dist,
                            "similarity_score": similarity_score,
                            "rank": rank,
                        }
                    )
        return retrieved_docs


def build_llm(model: str = "llama3-70b-8192") -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in the project .env before creating ChatGroq")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )


@dataclass
class RAGPipeline:
    embedder: EmbeddingManager
    vector_store: VectorStoreManager
    retriever: RAGRetriever


def build_pipeline() -> RAGPipeline:
    documents = load_all_text()
    if not documents:
        raise ValueError("No .txt files found in rag/data")

    chunks = split_docs(documents)
    embedder = EmbeddingManager()
    vector_store = VectorStoreManager()
    if vector_store.is_empty():
        embeddings = embedder.generate_embeddings([doc.page_content for doc in chunks])
        vector_store.add_documents(embeddings, chunks)
    retriever = RAGRetriever(embedder, vector_store)
    return RAGPipeline(embedder=embedder, vector_store=vector_store, retriever=retriever)
