import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


WAVE_ARG_MAP = {
    "uncertainty_shift": "wave_uncertainty",
    "divergence_shift": "wave_divergence",
    "risk_shift": "wave_risk",
    "revision_shift": "wave_revision",
}
QUERY_WAVE_ARG_MAP = {
    "uncertainty_shift": "query_wave_uncertainty",
    "divergence_shift": "query_wave_divergence",
    "risk_shift": "query_wave_risk",
    "revision_shift": "query_wave_revision",
}


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


def _wave_from_args(args: argparse.Namespace, query: bool) -> Optional[Dict[str, float]]:
    mapping = QUERY_WAVE_ARG_MAP if query else WAVE_ARG_MAP
    wave: Dict[str, float] = {}
    for key, attr in mapping.items():
        raw = getattr(args, attr)
        value = _bounded_float(raw, f"--{attr.replace('_', '-')}")
        if value is not None:
            wave[key] = value
    return wave or None


def _compute_friction_summary(
    metadata: Dict[str, object],
    query_tension: Optional[float],
    query_wave: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    tension_delta: Optional[float] = None
    wave_distance: Optional[float] = None

    doc_tension = metadata.get("tension")
    if query_tension is not None and isinstance(doc_tension, (int, float)):
        tension_delta = abs(float(query_tension) - float(doc_tension))
        tension_delta = max(0.0, min(1.0, tension_delta))

    doc_wave = metadata.get("wave")
    if isinstance(doc_wave, dict) and query_wave:
        shared = [key for key in query_wave.keys() if key in doc_wave]
        if shared:
            distance = [abs(float(query_wave[key]) - float(doc_wave[key])) for key in shared]
            wave_distance = float(np.mean(distance))
            wave_distance = max(0.0, min(1.0, wave_distance))

    if tension_delta is None and wave_distance is None:
        return None

    if tension_delta is not None and wave_distance is not None:
        friction = 0.5 * tension_delta + 0.5 * wave_distance
    elif tension_delta is not None:
        friction = tension_delta
    else:
        friction = wave_distance if wave_distance is not None else 0.0

    friction = float(max(0.0, min(1.0, friction)))
    return {
        "tension_delta": round(tension_delta, 4) if tension_delta is not None else None,
        "wave_distance": round(wave_distance, 4) if wave_distance is not None else None,
        "friction": round(friction, 4),
    }


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
            memory_kind=args.kind,
            wave=args.wave,
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
    parser.add_argument(
        "--kind",
        choices=sorted(Hippocampus.VALID_MEMORY_KINDS),
        default="note",
        help="Structured memory kind for ingestion",
    )
    parser.add_argument("--tension", type=float, help="Memory tension in [0,1] used at ingest time")
    parser.add_argument("--query-tension", type=float, help="Query tension in [0,1] for resonance reweighting")
    parser.add_argument(
        "--query-tension-mode",
        choices=["resonance", "conflict"],
        default="resonance",
        help="resonance=favor similar tension, conflict=favor high tension delta",
    )
    parser.add_argument("--tag", action="append", default=[], help="Repeat to attach tags to ingested memory")
    parser.add_argument("--wave-uncertainty", type=float, help="Memory wave uncertainty_shift in [0,1]")
    parser.add_argument("--wave-divergence", type=float, help="Memory wave divergence_shift in [0,1]")
    parser.add_argument("--wave-risk", type=float, help="Memory wave risk_shift in [0,1]")
    parser.add_argument("--wave-revision", type=float, help="Memory wave revision_shift in [0,1]")
    parser.add_argument(
        "--query-wave-mode",
        choices=["resonance", "conflict"],
        default="resonance",
        help="resonance=favor similar wave, conflict=favor high wave distance",
    )
    parser.add_argument("--query-wave-uncertainty", type=float, help="Query wave uncertainty_shift in [0,1]")
    parser.add_argument("--query-wave-divergence", type=float, help="Query wave divergence_shift in [0,1]")
    parser.add_argument("--query-wave-risk", type=float, help="Query wave risk_shift in [0,1]")
    parser.add_argument("--query-wave-revision", type=float, help="Query wave revision_shift in [0,1]")

    parser.add_argument("--json", action="store_true", help="Print retrieval result as JSON")
    parser.add_argument("--with-meta", action="store_true", help="Include stored metadata in output")
    parser.add_argument(
        "--friction-report",
        action="store_true",
        help="Show per-result friction summary from query vs memory metadata",
    )
    parser.add_argument(
        "--why-tonesoul",
        action="store_true",
        help="Print a short OpenClaw vs ToneSoul difference note and exit",
    )
    parser.add_argument(
        "--validate-structured",
        action="store_true",
        help="Run an in-process structured-memory validation scenario and exit",
    )
    parser.add_argument(
        "--validate-tension-replay",
        action="store_true",
        help="Run high/low tension replay validation and exit with non-zero on failure",
    )
    return parser


def _print_profile_note() -> None:
    _emit("OpenClaw vs ToneSoul")
    _emit("- openclaw: FAISS + BM25 + RRF + time decay (baseline memory retrieval)")
    _emit("- tonesoul: baseline + tension signal (resonance/conflict) + structured wave rerank")
    _emit("- design goal: keep baseline simple while making disagreement/context pressure visible")


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "tonesoul":
        if args.tension is None and (args.learn or args.memory_file):
            args.tension = 0.70
        if args.wave is None and (args.learn or args.memory_file):
            t = args.tension if args.tension is not None else 0.70
            args.wave = {
                "uncertainty_shift": min(1.0, 0.45 + 0.35 * t),
                "divergence_shift": min(1.0, 0.35 + 0.40 * t),
                "risk_shift": min(1.0, 0.30 + 0.55 * t),
                "revision_shift": min(1.0, 0.25 + 0.30 * t),
            }
        if args.query_wave is None and args.query_tension is not None:
            t = args.query_tension
            args.query_wave = {
                "uncertainty_shift": min(1.0, 0.40 + 0.35 * t),
                "divergence_shift": min(1.0, 0.30 + 0.45 * t),
                "risk_shift": min(1.0, 0.30 + 0.55 * t),
                "revision_shift": min(1.0, 0.20 + 0.35 * t),
            }
        if "tonesoul" not in args.tag:
            args.tag.append("tonesoul")
    else:
        args.query_tension = None
        args.query_wave = None
        args.query_tension_mode = "resonance"
        args.query_wave_mode = "resonance"


def _run_structured_validation() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="openclaw_structured_") as tmp_dir:
        hippo = Hippocampus(db_path=tmp_dir, embedder=HashEmbedding())
        low = hippo.memorize(
            content="deployment decision memory",
            source_file="validation",
            memory_kind="decision",
            tension=0.25,
            wave={
                "uncertainty_shift": 0.20,
                "divergence_shift": 0.20,
                "risk_shift": 0.20,
                "revision_shift": 0.20,
            },
            tags=["validation", "low"],
        )
        high = hippo.memorize(
            content="deployment decision memory",
            source_file="validation",
            memory_kind="decision",
            tension=0.90,
            wave={
                "uncertainty_shift": 0.90,
                "divergence_shift": 0.85,
                "risk_shift": 0.95,
                "revision_shift": 0.80,
            },
            tags=["validation", "high"],
        )
        result = hippo.recall(
            query_text="deployment decision memory",
            top_k=2,
            query_tension=0.9,
            query_wave={
                "uncertainty_shift": 0.90,
                "divergence_shift": 0.90,
                "risk_shift": 0.95,
                "revision_shift": 0.85,
            },
        )

        ranked_ids = [item.doc_id for item in result]
        if ranked_ids and ranked_ids[0] == high and low in ranked_ids:
            _emit("structured validation: PASS")
            _emit("expected high-wave memory ranked first.")
        else:
            _emit("structured validation: FAIL")
            _emit(f"ranked_ids={ranked_ids}")


def _run_tension_replay_validation() -> bool:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="openclaw_tension_replay_") as tmp_dir:
        hippo = Hippocampus(db_path=tmp_dir, embedder=HashEmbedding())
        obedience_id = hippo.memorize(
            content="core conflict memory record",
            source_file="obedience_memory",
            memory_kind="decision",
            tension=0.20,
            tags=["validation", "obedience"],
        )
        boundary_id = hippo.memorize(
            content="core conflict memory record",
            source_file="boundary_memory",
            memory_kind="decision",
            tension=0.90,
            tags=["validation", "boundary"],
        )

        high_query = hippo.recall(
            query_text="core conflict memory record",
            top_k=2,
            query_tension=0.90,
            query_tension_mode="resonance",
        )
        low_query = hippo.recall(
            query_text="core conflict memory record",
            top_k=2,
            query_tension=0.10,
            query_tension_mode="resonance",
        )

        high_top = high_query[0].doc_id if high_query else None
        low_top = low_query[0].doc_id if low_query else None
        ok = high_top == boundary_id and low_top == obedience_id

        _emit("tension replay validation: PASS" if ok else "tension replay validation: FAIL")
        _emit(f"high_tension_top={high_top} expected={boundary_id}")
        _emit(f"low_tension_top={low_top} expected={obedience_id}")
        return ok


def main() -> None:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if args.why_tonesoul:
        _print_profile_note()
        return
    args.wave = _wave_from_args(args, query=False)
    args.query_wave = _wave_from_args(args, query=True)
    if args.validate_tension_replay:
        if not _run_tension_replay_validation():
            sys.exit(1)
        return
    if args.validate_structured:
        _run_structured_validation()
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
        query_tension_mode=args.query_tension_mode,
        query_wave=args.query_wave,
        query_wave_mode=args.query_wave_mode,
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
        if args.with_meta:
            for item, res in zip(payload, results):
                item["metadata"] = res.metadata
        if args.friction_report:
            for item, res in zip(payload, results):
                item["friction"] = _compute_friction_summary(
                    res.metadata, args.query_tension, args.query_wave
                )
        _emit(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    _emit(f"query: {query_text}")
    _emit("=" * 50)
    for res in results:
        _emit(f"[{res.rank}] source={res.source_file} score={res.score:.4f}")
        if args.with_meta:
            summary = {
                "kind": res.metadata.get("kind"),
                "tension": res.metadata.get("tension"),
                "tags": res.metadata.get("tags", []),
                "wave": res.metadata.get("wave"),
            }
            _emit(f"meta: {json.dumps(summary, ensure_ascii=True)}")
        if args.friction_report:
            friction = _compute_friction_summary(res.metadata, args.query_tension, args.query_wave)
            if friction is not None:
                _emit(f"friction: {json.dumps(friction, ensure_ascii=True)}")
        _emit(res.content)
        _emit("-" * 50)


if __name__ == "__main__":
    main()
