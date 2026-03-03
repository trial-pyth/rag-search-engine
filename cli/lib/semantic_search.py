from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from lib.search_utils import load_movies

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError(
            "Missing dependency `numpy`. Install project dependencies (e.g. `uv sync`)."
        )
    return np


class SemanticSearch:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency `sentence-transformers`. Install project dependencies (e.g. `uv sync`)."
            ) from e

        self.model = SentenceTransformer(model_name)
        self.embeddings: Any = None
        self.documents: list[dict[str, Any]] = []
        self.document_map: dict[Any, dict[str, Any]] = {}
        self.embeddings_path = Path("cache/movie_embeddings.npy")

    def build_embeddings(self, documents: list[dict[str, Any]]):
        np_mod = _require_numpy()
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np_mod.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str, Any]]):
        np_mod = _require_numpy()
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if self.embeddings_path.exists():
            embeddings = np_mod.load(self.embeddings_path)
            self.embeddings = embeddings
            if len(self.documents) == len(embeddings):
                return embeddings
        return self.build_embeddings(documents)


    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int):
        embeddings = self.embeddings
        documents = self.documents
        if embeddings is None:
            raise ValueError("No embeddings are loaded. Call load_or_create_embeddings() first.")
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Embeddings/documents length mismatch: {len(embeddings)} != {len(documents)}"
            )
        
        qry_emb = self.generate_embedding(query)
        similarities = []
        for doc_emb, doc in zip(embeddings, documents):
            similarity = cosine_similarity(qry_emb, doc_emb)
            similarities.append((float(similarity), doc))

        similarities.sort(key = lambda x: x[0], reverse= True)
        res = []
        for sc, doc in similarities[:limit]:
            res.append({
                'score': sc, 
                'title': doc['title'], 
                'description': doc['description']
            })
        return res

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__(model_name=model_name)
        self.chunk_embeddings: Any = None
        self.chunk_metadata: list[dict[str, Any]] = []
        self.chunk_metadata_path = Path("cache/chunk_metadata.json")
        self.chunk_embeddings_path = Path("cache/chunk_embeddings.npy")

    def build_chunk_embeddings(self, documents: list[dict[str, Any]]):
        np_mod = _require_numpy()
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        
        all_chunks = []
        chunk_metadata = []

        for midx, doc in enumerate(documents):
            description = (doc.get("description") or "").strip()
            if not description:
                continue
            _chunks = semantic_chunking(description, overlap=1, max_chunk_size=4)
            all_chunks.extend(_chunks)
            for cidx in range(len(_chunks)):
                chunk_metadata.append({
                    "movie_idx": midx,
                    "movie_id": doc.get("id"),
                    "chunk_idx": cidx,
                    "total_chunks": len(_chunks),
                    "text": _chunks[cidx],
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        self.chunk_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np_mod.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, 'w', encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict[str, Any]]):
        np_mod = _require_numpy()
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            chunk_embeddings = np_mod.load(self.chunk_embeddings_path)
            self.chunk_embeddings = chunk_embeddings

            with open(self.chunk_metadata_path, 'r', encoding="utf-8") as f:
                payload = json.load(f)
                self.chunk_metadata = payload.get("chunks", [])
            return chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        chunk_embeddings = self.chunk_embeddings
        if chunk_embeddings is None:
            raise ValueError(
                "No chunk embeddings are loaded. Call load_or_create_chunk_embeddings() first."
            )
        if len(chunk_embeddings) != len(self.chunk_metadata):
            raise ValueError(
                "Chunk embeddings/metadata length mismatch: "
                f"{len(chunk_embeddings)} != {len(self.chunk_metadata)}"
            )

        query_emb = self.generate_embedding(query)

        best_by_movie_id: dict[Any, tuple[float, dict[str, Any]]] = {}
        for idx in range(len(chunk_embeddings)):
            chunk_embedding = chunk_embeddings[idx]
            metadata = self.chunk_metadata[idx]
            movie_id = metadata.get("movie_id")
            if movie_id is None:
                continue

            sim = cosine_similarity(query_emb, chunk_embedding)
            current_best = best_by_movie_id.get(movie_id)
            if current_best is None or sim > current_best[0]:
                best_by_movie_id[movie_id] = (float(sim), metadata)

        movies_scores_sorted = sorted(
            best_by_movie_id.items(), key=lambda item: item[1][0], reverse=True
        )

        res = []
        for movie_id, (score, metadata) in movies_scores_sorted[:limit]:
            doc = self.document_map[movie_id]
            res.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc.get("description", "")[:200],
                    "score": round(score, 4),
                    "metadata": {
                        "chunk_idx": metadata.get("chunk_idx"),
                        "chunk_text": metadata.get("text", "")[:200],
                        "total_chunks": metadata.get("total_chunks"),
                    },
                }
            )
        return res


def search_chunked(query: str, limit: int = 5):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, item in enumerate(results, start=1):
        print(f"\n{i}. {item['title']} (score: {item['score']:.4f})")
        print(f"   {item['description']}...")



def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")

def chunk_text_semantic(text, overlap = 0, max_chunk_size = 4):
    chunks = semantic_chunking(text, overlap ,max_chunk_size)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")        

def semantic_chunking(text, overlap = 0 , max_chunk_size = 4):
    text = text.strip()
    if not text:
        return [] 
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0].endswith(('!', '.', '?')) :
        pass
    chunks = []
    step_size = max_chunk_size - overlap
    sentences = [s.strip() for s in sentences if s]
    for i in range(0,len(sentences), step_size):
        chunk_sentences = sentences[i:i+max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def fixed_sized_chunking(text, overlap,  chunk_size=200):
    words = text.split()
    chunks = [] 
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks

def chunk_text(text, overlap ,chunk_size=200):
    chunks = fixed_sized_chunking(text, overlap,int(chunk_size))
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def search(query, limit = 5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    search_results = ss.search(query, limit)
    for idx, res in enumerate(search_results):
        print(f"{idx}. {res['title']} (score: {res['score']:.4f})")
        print(res['description'][:100])

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss =SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    np_mod = _require_numpy()
    dot_product = np_mod.dot(vec1, vec2)
    norm1 = np_mod.linalg.norm(vec1)
    norm2 = np_mod.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
