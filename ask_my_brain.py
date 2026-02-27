import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from openclaw_memory.hippocampus import Hippocampus


class HashEmbedding:
    """Deterministic offline embedder for local testing and no-network usage."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _emit(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        fallback = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(fallback)


def _bounded_float(value: Optional[float], flag_name: str) -> Optional[float]:
    if value is None:
        return None
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{flag_name} must be between 0.0 and 1.0")
    return value


def _read_memory_chunks(file_path: Path) -> List[str]:
    text = None
    for encoding in ("utf-8", "utf-16", "cp950"):
        try:
            text = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"failed to decode {file_path}")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks and text.strip():
        return [text.strip()]
    return chunks


def _ingest_cli_memories(hippo: Hippocampus, args: argparse.Namespace) -> int:
    entries: List[Tuple[str, str]] = []
    if args.learn:
        entries.append((args.learn.strip(), args.source_file))

    if args.memory_file:
        file_path = Path(args.memory_file)
        if not file_path.exists():
            raise FileNotFoundError(f"memory file not found: {file_path}")
        for chunk in _read_memory_chunks(file_path):
            entries.append((chunk, file_path.name))

    inserted = 0
    for content, source_file in entries:
        if not content:
            continue
        doc_id = hippo.memorize(
            content=content,
            source_file=source_file,
            origin=args.origin,
            tension=args.tension,
            tags=args.tag,
        )
        inserted += 1
        _emit(f"ingested doc_id={doc_id} source={source_file}")
    return inserted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw memory CLI")
    parser.add_argument("query", nargs="*", help="Query text to retrieve relevant memories")
    parser.add_argument("--db-path", type=str, default="./memory_base", help="Memory database path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recall results")
    parser.add_argument(
        "--profile",
        choices=["openclaw", "tonesoul"],
        default="openclaw",
        help="openclaw=baseline hybrid retrieval, tonesoul=tension-aware defaults",
    )

    parser.add_argument("--learn", type=str, help="Ingest one memory sentence/paragraph")
    parser.add_argument("--memory-file", type=str, help="Ingest memories from a UTF-8 text/markdown file")
    parser.add_argument("--source-file", type=str, default="cli_ingestion", help="source_file metadata value")
    parser.add_argument("--origin", type=str, default="agent_consolidation", help="origin metadata value")
    parser.add_argument("--tension", type=float, help="Memory tension in [0,1] used at ingest time")
    parser.add_argument("--query-tension", type=float, help="Query tension in [0,1] for resonance reweighting")
    parser.add_argument("--tag", action="append", default=[], help="Repeat to attach tags to ingested memory")

    parser.add_argument("--json", action="store_true", help="Print retrieval result as JSON")
    parser.add_argument(
        "--why-tonesoul",
        action="store_true",
        help="Print a short OpenClaw vs ToneSoul difference note and exit",
    )
    return parser


def _print_profile_note() -> None:
    _emit("OpenClaw vs ToneSoul")
    _emit("- openclaw: FAISS + BM25 + RRF + time decay (baseline memory retrieval)")
    _emit("- tonesoul: baseline + optional tension metadata and query-time resonance rerank")
    _emit("- design goal: keep baseline simple while making disagreement/context pressure visible")


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "tonesoul":
        if args.tension is None and (args.learn or args.memory_file):
            args.tension = 0.70
        if "tonesoul" not in args.tag:
            args.tag.append("tonesoul")
    else:
        args.query_tension = None


def main() -> None:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if args.why_tonesoul:
        _print_profile_note()
        return

    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    try:
        args.tension = _bounded_float(args.tension, "--tension")
        args.query_tension = _bounded_float(args.query_tension, "--query-tension")
    except ValueError as exc:
        parser.error(str(exc))
    _apply_profile_defaults(args)

    hippo = Hippocampus(db_path=args.db_path, embedder=HashEmbedding())

    inserted = _ingest_cli_memories(hippo, args)
    if inserted > 0 and not args.query:
        _emit(f"ingested {inserted} memory item(s).")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    query_text = " ".join(args.query).strip()
    results = hippo.recall(
        query_text=query_text,
        top_k=args.top_k,
        query_tension=args.query_tension,
    )
    if not results:
        _emit("no memories found.")
        return

    if args.json:
        payload = [
            {
                "rank": res.rank,
                "doc_id": res.doc_id,
                "source_file": res.source_file,
                "score": res.score,
                "content": res.content,
            }
            for res in results
        ]
        _emit(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    _emit(f"query: {query_text}")
    _emit("=" * 50)
    for res in results:
        _emit(f"[{res.rank}] source={res.source_file} score={res.score:.4f}")
        _emit(res.content)
        _emit("-" * 50)


if __name__ == "__main__":
    main()
