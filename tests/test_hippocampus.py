import os
import json
import uuid
import numpy as np
from datetime import datetime
import pytest

from openclaw_memory.hippocampus import Hippocampus
from openclaw_memory.embeddings import MockEmbedding

@pytest.fixture
def mock_db_path(tmp_path):
    path = tmp_path / "mock_memory_base"
    path.mkdir()
    
    # Generate some fake data
    dim = 384
    meta_file = path / "tonesoul_metadata.jsonl"
    
    # Write metadata
    now = datetime.utcnow().isoformat()
    old_time = datetime(2000, 1, 1).isoformat()
    
    docs = [
        {"id": "1", "source_file": "doc1.md", "content": "The lobster has a decentralized nervous system.", "ingested_at": now},
        {"id": "2", "source_file": "doc2.md", "content": "Memory is stored via vector embeddings.", "ingested_at": now},
        {"id": "3", "source_file": "doc3.md", "content": "Pinecone is a cloud database.", "ingested_at": old_time} # Old memory to test decay
    ]
    
    with open(meta_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
            
    # Write FAISS Index
    import faiss
    index = faiss.IndexFlatIP(dim)
    # Vectors close to zero, doc 1 is slightly different for keyword testing mostly
    vecs = np.zeros((3, dim), dtype=np.float32)
    # Give doc1 a specific vector signature
    vecs[0][0] = 1.0
    vecs[1][1] = 1.0
    vecs[2][2] = 1.0
    index.add(vecs)
    faiss.write_index(index, str(path / "tonesoul_cognitive.index"))
    
    return str(path)

def test_hippocampus_initialization(mock_db_path):
    hippo = Hippocampus(db_path=mock_db_path)
    assert len(hippo.metadata) == 3
    assert hippo.index.ntotal == 3
    assert hippo.bm25 is not None

def test_time_decay(mock_db_path):
    hippo = Hippocampus(db_path=mock_db_path)
    # Give it a base score of 1.0
    recent_score = hippo._apply_time_decay(1.0, datetime.utcnow().isoformat())
    old_score = hippo._apply_time_decay(1.0, datetime(2000, 1, 1).isoformat())
    
    # The old score should be drastically decayed compared to the recent one
    assert old_score < recent_score
    assert recent_score >= 0.99 # Should technically be 1.0 since it's today

def test_hybrid_recall(mock_db_path):
    mock_embedder = MockEmbedding(dimension=384)
    hippo = Hippocampus(db_path=mock_db_path, embedder=mock_embedder)
    
    # Mock embedding returns exactly zeros, so FAISS scores will be 0.
    # Therefore, BM25 Keyword search will dominate the rank fusion.
    # Query for "lobster" should heavily favor doc 1.
    results = hippo.recall("lobster nervous system")
    
    assert len(results) > 0
    assert results[0].doc_id == "1"
    assert "lobster" in results[0].content
