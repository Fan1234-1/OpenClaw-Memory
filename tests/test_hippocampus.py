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

def test_path_traversal_rejected():
    """Verify that Hippocampus rejects db_path containing path traversal."""
    with pytest.raises(ValueError, match="path traversal"):
        Hippocampus(db_path="../../../etc/passwd")

def test_memorize_persists_tension_and_tags(tmp_path):
    path = tmp_path / "fresh_memory"
    hippo = Hippocampus(db_path=str(path), embedder=MockEmbedding(dimension=384))

    doc_id = hippo.memorize(
        content="Tension-bearing memory",
        source_file="unit_test",
        memory_kind="decision",
        tension=0.72,
        tags=["tension", "unit-test"],
        wave={
            "uncertainty_shift": 0.4,
            "divergence_shift": 0.5,
            "risk_shift": 0.8,
            "revision_shift": 0.3,
        },
    )

    assert doc_id
    assert hippo.metadata[-1]["kind"] == "decision"
    assert hippo.metadata[-1]["tension"] == pytest.approx(0.72)
    assert hippo.metadata[-1]["tags"] == ["tension", "unit-test"]
    assert hippo.metadata[-1]["wave"]["risk_shift"] == pytest.approx(0.8)

def test_query_tension_resonance_reorders_results(tmp_path):
    path = tmp_path / "resonance_memory"
    hippo = Hippocampus(db_path=str(path), embedder=MockEmbedding(dimension=384))

    low_tension_id = hippo.memorize(
        content="shared conflict memory",
        source_file="low_tension",
        tension=0.1
    )
    high_tension_id = hippo.memorize(
        content="shared conflict memory",
        source_file="high_tension",
        tension=0.9
    )

    baseline = hippo.recall("shared conflict memory", top_k=2)
    resonant = hippo.recall("shared conflict memory", top_k=2, query_tension=0.9)

    assert {item.doc_id for item in baseline} == {low_tension_id, high_tension_id}
    assert {item.doc_id for item in resonant} == {low_tension_id, high_tension_id}
    assert resonant[0].doc_id == high_tension_id

def test_query_tension_conflict_reorders_results(tmp_path):
    path = tmp_path / "conflict_memory"
    hippo = Hippocampus(db_path=str(path), embedder=MockEmbedding(dimension=384))

    low_tension_id = hippo.memorize(
        content="conflict target memory",
        source_file="low_tension",
        tension=0.1,
    )
    high_tension_id = hippo.memorize(
        content="conflict target memory",
        source_file="high_tension",
        tension=0.9,
    )

    conflict = hippo.recall(
        "conflict target memory",
        top_k=2,
        query_tension=0.9,
        query_tension_mode="conflict",
    )

    assert {item.doc_id for item in conflict} == {low_tension_id, high_tension_id}
    assert conflict[0].doc_id == low_tension_id

def test_query_wave_resonance_reorders_results(tmp_path):
    path = tmp_path / "wave_memory"
    hippo = Hippocampus(db_path=str(path), embedder=MockEmbedding(dimension=384))

    low_wave_id = hippo.memorize(
        content="structured memory target",
        source_file="low_wave",
        wave={
            "uncertainty_shift": 0.1,
            "divergence_shift": 0.2,
            "risk_shift": 0.2,
            "revision_shift": 0.2,
        },
    )
    high_wave_id = hippo.memorize(
        content="structured memory target",
        source_file="high_wave",
        wave={
            "uncertainty_shift": 0.9,
            "divergence_shift": 0.9,
            "risk_shift": 0.95,
            "revision_shift": 0.85,
        },
    )

    baseline = hippo.recall("structured memory target", top_k=2)
    resonant = hippo.recall(
        "structured memory target",
        top_k=2,
        query_wave={
            "uncertainty_shift": 0.9,
            "divergence_shift": 0.85,
            "risk_shift": 0.95,
            "revision_shift": 0.9,
        },
    )

    assert {item.doc_id for item in baseline} == {low_wave_id, high_wave_id}
    assert {item.doc_id for item in resonant} == {low_wave_id, high_wave_id}
    assert resonant[0].doc_id == high_wave_id

def test_query_wave_conflict_reorders_results(tmp_path):
    path = tmp_path / "wave_conflict_memory"
    hippo = Hippocampus(db_path=str(path), embedder=MockEmbedding(dimension=384))

    low_wave_id = hippo.memorize(
        content="structured wave conflict target",
        source_file="low_wave",
        wave={
            "uncertainty_shift": 0.2,
            "divergence_shift": 0.2,
            "risk_shift": 0.2,
            "revision_shift": 0.2,
        },
    )
    high_wave_id = hippo.memorize(
        content="structured wave conflict target",
        source_file="high_wave",
        wave={
            "uncertainty_shift": 0.9,
            "divergence_shift": 0.9,
            "risk_shift": 0.95,
            "revision_shift": 0.85,
        },
    )

    conflict = hippo.recall(
        "structured wave conflict target",
        top_k=2,
        query_wave={
            "uncertainty_shift": 0.9,
            "divergence_shift": 0.9,
            "risk_shift": 0.95,
            "revision_shift": 0.9,
        },
        query_wave_mode="conflict",
    )

    assert {item.doc_id for item in conflict} == {low_wave_id, high_wave_id}
    assert conflict[0].doc_id == low_wave_id

