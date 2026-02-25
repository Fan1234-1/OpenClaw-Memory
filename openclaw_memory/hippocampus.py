import os
import json
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Any, Optional
from openclaw_memory.embeddings import BaseEmbedding

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

class MemoryResult:
    def __init__(self, doc_id: str, content: str, source_file: str, score: float, rank: int):
        self.doc_id = doc_id
        self.content = content
        self.source_file = source_file
        self.score = score
        self.rank = rank

class Hippocampus:
    """
    ToneSoul's Hybrid RAG Memory Retriever.
    Combines FAISS Vector Search with time-decay and BM25 Keyword Search.
    Inspired by 'Personal AI Memory'.
    """
    def __init__(self, db_path: str = "memory_base", embedder: Optional[BaseEmbedding] = None):
        # Validate db_path to prevent path traversal attacks
        normalized = os.path.normpath(db_path)
        if ".." in normalized.split(os.sep):
            raise ValueError(f"Invalid db_path: path traversal detected in '{db_path}'")
        self.db_path = os.path.abspath(db_path)
        self.index_file = os.path.join(self.db_path, "tonesoul_cognitive.index")
        self.meta_file = os.path.join(self.db_path, "tonesoul_metadata.jsonl")
        
        self.embedder = embedder
        self.index = None
        self.metadata = []
        self.bm25 = None
        
        self._load_db()

    def _load_db(self):
        if not os.path.exists(self.index_file) or not os.path.exists(self.meta_file):
            print("Memory Base not found. Please run ingest_ancestral_memory.py first.")
            return

        self.index = faiss.read_index(self.index_file)
        
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        
        # Initialize BM25 if available
        if BM25Okapi is not None and self.metadata:
            tokenized_corpus = [doc['content'].split(" ") for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def _apply_time_decay(self, base_score: float, ingested_at: str, half_life_days: float = 69.0) -> float:
        """Applies exponential time decay: score = score * exp(-lambda * days_old)"""
        try:
            record_time = datetime.fromisoformat(ingested_at)
            days_old = (datetime.utcnow() - record_time).days
            days_old = max(0, days_old)
            decay_rate = 0.01 # Approx for half-life of 69 days
            return float(base_score * np.exp(-decay_rate * days_old))
        except Exception:
            return base_score

    def search_vectors(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
            
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            raw_score = distances[0][i]
            decayed_score = self._apply_time_decay(raw_score, meta.get("ingested_at", datetime.utcnow().isoformat()))
            
            results.append({
                "doc": meta,
                "score": decayed_score,
                "type": "vector"
            })
            
        # Sort desc by decayed score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_keywords(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
            
        tokenized_query = query_text.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            results.append({
                "doc": self.metadata[idx],
                "score": score,
                "type": "keyword"
            })
            
        return results

    def recall(self, query_text: str, query_vector: Optional[np.ndarray] = None, top_k: int = 5) -> List[MemoryResult]:
        """
        Main retrieval function using Reciprocal Rank Fusion (RRF).
        """
        if query_vector is None:
            if self.embedder is None:
                raise ValueError("hippocampus requires either a query_vector or an initialized embedder.")
            query_vector = self.embedder.encode(query_text)
            
        vec_results = self.search_vectors(query_vector, top_k=20)
        kw_results = self.search_keywords(query_text, top_k=20)
        
        # RRF Fusion (k=60 is standard)
        rrf_k = 60
        fusion_scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}
        
        # Process Vector Ranks
        for rank, item in enumerate(vec_results):
            doc_id = item["doc"]["id"]
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0.0
                doc_map[doc_id] = item["doc"]
            fusion_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        # Process Keyword Ranks
        for rank, item in enumerate(kw_results):
            doc_id = item["doc"]["id"]
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0.0
                doc_map[doc_id] = item["doc"]
            fusion_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        # Sort and return top_k
        sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_map[doc_id]
            final_results.append(
                MemoryResult(
                    doc_id=doc_id,
                    content=doc["content"],
                    source_file=doc["source_file"],
                    score=score,
                    rank=rank + 1
                )
            )
            
        return final_results
