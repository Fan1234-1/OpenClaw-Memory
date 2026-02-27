import os
import json
import uuid
import numpy as np
import faiss
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from openclaw_memory.embeddings import BaseEmbedding

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

class MemoryResult:
    def __init__(
        self,
        doc_id: str,
        content: str,
        source_file: str,
        score: float,
        rank: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.doc_id = doc_id
        self.content = content
        self.source_file = source_file
        self.score = score
        self.rank = rank
        self.metadata = metadata or {}

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
        os.makedirs(self.db_path, exist_ok=True)
        
        if not os.path.exists(self.index_file):
            print("Memory Base not found. Initializing empty FAISS index.")
            # Default dimension to 384 for all-MiniLM-L6-v2, can be overridden if embedder provides dimension
            dim = 384
            self.index = faiss.IndexFlatIP(dim)
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                pass # Create empty file
        else:
            self.index = faiss.read_index(self.index_file)
            
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.metadata.append(json.loads(line))
        
        self._rebuild_bm25()

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
        
    def _rebuild_bm25(self):
        if BM25Okapi is not None and self.metadata:
            tokenized_corpus = [doc['content'].split(" ") for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    @classmethod
    def _validate_memory_kind(cls, memory_kind: str) -> str:
        kind = str(memory_kind).strip().lower()
        if not kind:
            raise ValueError("memory_kind cannot be empty")
        if kind not in cls.VALID_MEMORY_KINDS:
            allowed = ", ".join(sorted(cls.VALID_MEMORY_KINDS))
            raise ValueError(f"memory_kind must be one of: {allowed}")
        return kind

    @classmethod
    def _validate_wave(cls, wave: Optional[Dict[str, Any]], field_name: str) -> Optional[Dict[str, float]]:
        if wave is None:
            return None
        if not isinstance(wave, dict):
            raise ValueError(f"{field_name} must be a dict")

        normalized: Dict[str, float] = {}
        for key, value in wave.items():
            if key not in cls.WAVE_KEYS:
                allowed = ", ".join(cls.WAVE_KEYS)
                raise ValueError(f"{field_name} contains unknown key '{key}'. Allowed keys: {allowed}")
            if value is None:
                continue
            numeric = float(value)
            if not (0.0 <= numeric <= 1.0):
                raise ValueError(f"{field_name}.{key} must be between 0.0 and 1.0")
            normalized[key] = numeric

        return normalized or None

    def memorize(
        self,
        content: str,
        source_file: str = "runtime_experience",
        origin: str = "agent_consolidation",
        tension: Optional[float] = None,
        tags: Optional[List[str]] = None,
        memory_kind: str = "note",
        wave: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Dynamically ingests a new memory chunk into the current index and persistent storage.
        """
        if self.embedder is None:
            raise ValueError("Cannot memorize: no embedder provided to Hippocampus.")
        if tension is not None and not (0.0 <= float(tension) <= 1.0):
            raise ValueError("tension must be between 0.0 and 1.0")
        normalized_kind = self._validate_memory_kind(memory_kind)
        normalized_wave = self._validate_wave(wave, "wave")
            
        doc_id = str(uuid.uuid4())
        vector = self.embedder.encode(content)
        
        # 1. Update FAISS
        vector_matrix = np.array([vector], dtype=np.float32)
        self.index.add(vector_matrix)
        faiss.write_index(self.index, self.index_file)
        
        # 2. Update Metadata
        meta = {
            "id": doc_id,
            "source_file": source_file,
            "content": content,
            "ingested_at": self._utcnow_iso(),
            "origin": origin,
            "kind": normalized_kind,
        }
        if tension is not None:
            meta["tension"] = float(tension)
        if tags:
            meta["tags"] = [str(tag) for tag in tags if str(tag).strip()]
        if normalized_wave is not None:
            meta["wave"] = normalized_wave
        self.metadata.append(meta)
        
        with open(self.meta_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(meta) + "\n")
            
        # 3. Rebuild BM25
        self._rebuild_bm25()
        
        return doc_id

    @staticmethod
    def _apply_tension_resonance(base_score: float, doc: Dict[str, Any], query_tension: Optional[float]) -> float:
        return Hippocampus._apply_tension_signal(base_score, doc, query_tension, mode="resonance")

    @staticmethod
    def _apply_tension_signal(
        base_score: float,
        doc: Dict[str, Any],
        query_tension: Optional[float],
        mode: str = "resonance",
    ) -> float:
        """
        Light reweighting over RRF score.
        mode="resonance" favors similar tension.
        mode="conflict" favors high tension difference.
        """
        if query_tension is None:
            return base_score
        if not (0.0 <= query_tension <= 1.0):
            return base_score

        doc_tension = doc.get("tension")
        if doc_tension is None:
            return base_score

        try:
            doc_tension = float(doc_tension)
        except (TypeError, ValueError):
            return base_score
        if not (0.0 <= doc_tension <= 1.0):
            return base_score

        delta = abs(query_tension - doc_tension)
        delta = max(0.0, min(1.0, delta))
        if mode == "conflict":
            signal = delta
        else:
            signal = 1.0 - delta
        return float(base_score * (1.0 + 0.20 * signal))

    @classmethod
    def _apply_wave_resonance(
        cls,
        base_score: float,
        doc: Dict[str, Any],
        query_wave: Optional[Dict[str, float]],
        mode: str = "resonance",
    ) -> float:
        """
        Reweight by "AI-meaningful" wave similarity.
        We only compare shared dimensions to stay backward compatible.
        """
        if query_wave is None:
            return base_score

        doc_wave_raw = doc.get("wave")
        if not isinstance(doc_wave_raw, dict):
            return base_score

        try:
            doc_wave = cls._validate_wave(doc_wave_raw, "doc.wave")
        except ValueError:
            return base_score
        if doc_wave is None:
            return base_score

        shared_keys = [key for key in cls.WAVE_KEYS if key in query_wave and key in doc_wave]
        if not shared_keys:
            return base_score

        distance = float(np.mean([abs(query_wave[key] - doc_wave[key]) for key in shared_keys]))
        distance = max(0.0, min(1.0, distance))
        if mode == "conflict":
            signal = distance
        else:
            signal = 1.0 - distance
        return float(base_score * (1.0 + 0.25 * signal))

    def _apply_time_decay(self, base_score: float, ingested_at: str, half_life_days: float = 69.0) -> float:
        """Applies exponential time decay: score = score * exp(-lambda * days_old)"""
        try:
            record_time = datetime.fromisoformat(str(ingested_at).replace("Z", "+00:00"))
            if record_time.tzinfo is None:
                record_time = record_time.replace(tzinfo=timezone.utc)
            days_old = (datetime.now(timezone.utc) - record_time).days
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
            decayed_score = self._apply_time_decay(
                raw_score,
                meta.get("ingested_at", self._utcnow_iso()),
            )
            
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

    def recall(
        self,
        query_text: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        query_tension: Optional[float] = None,
        query_tension_mode: str = "resonance",
        query_wave: Optional[Dict[str, Any]] = None,
        query_wave_mode: str = "resonance",
    ) -> List[MemoryResult]:
        """
        Main retrieval function using Reciprocal Rank Fusion (RRF).
        """
        if query_vector is None:
            if self.embedder is None:
                raise ValueError("hippocampus requires either a query_vector or an initialized embedder.")
            query_vector = self.embedder.encode(query_text)
        if query_tension_mode not in {"resonance", "conflict"}:
            raise ValueError("query_tension_mode must be 'resonance' or 'conflict'")
        if query_wave_mode not in {"resonance", "conflict"}:
            raise ValueError("query_wave_mode must be 'resonance' or 'conflict'")
        normalized_query_wave = self._validate_wave(query_wave, "query_wave")
            
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
        adjusted_scores: Dict[str, float] = {}
        for doc_id, score in fusion_scores.items():
            adjusted = self._apply_tension_signal(score, doc_map[doc_id], query_tension, mode=query_tension_mode)
            adjusted = self._apply_wave_resonance(
                adjusted,
                doc_map[doc_id],
                normalized_query_wave,
                mode=query_wave_mode,
            )
            adjusted_scores[doc_id] = adjusted

        sorted_docs = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = doc_map[doc_id]
            final_results.append(
                MemoryResult(
                    doc_id=doc_id,
                    content=doc["content"],
                    source_file=doc["source_file"],
                    score=score,
                    rank=rank + 1,
                    metadata=doc,
                )
            )
            
        return final_results
    VALID_MEMORY_KINDS = {
        "note",
        "fact",
        "decision",
        "constraint",
        "reflection",
        "incident",
        "plan",
    }
    WAVE_KEYS: Tuple[str, ...] = (
        "uncertainty_shift",
        "divergence_shift",
        "risk_shift",
        "revision_shift",
    )
