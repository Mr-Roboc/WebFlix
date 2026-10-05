#!/usr/bin/env python3
"""JSON search adapter for the Go WebFlix TUI. Prints one JSON object to stdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from typing import Any

CLI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CLI_DIR)

if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

os.chdir(REPO_ROOT)

from lib.search_utils import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)


@contextmanager
def stdout_to_stderr() -> Any:
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return float(obj)
    return obj


def snippet(text: str, length: int = DOCUMENT_PREVIEW_LENGTH) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) > length:
        return text[:length] + "..."
    return text


def normalize_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    title = item.get("title") or item.get("doc_title") or ""
    document = item.get("document") or item.get("description") or ""
    score = item.get("score")
    if score is None:
        score = item.get(
            "cross_encoder_score",
            item.get("rerank_score", item.get("hybrid_score", item.get("rrf_score", 0.0))),
        )
    out: dict[str, Any] = {
        "id": item.get("id", 0),
        "title": title,
        "document": snippet(str(document)),
        "score": float(score or 0.0),
        "rank": rank,
    }
    for key in (
        "bm25_score",
        "sem_score",
        "hybrid_score",
        "rrf_score",
        "rerank_score",
        "batch_rank",
        "cross_encoder_score",
        "bm25_rank",
        "semantic_rank",
    ):
        if key in item and item[key] is not None:
            out[key] = jsonable(item[key])
    return out


def emit(payload: dict[str, Any], exit_code: int = 0) -> None:
    json.dump(jsonable(payload), sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    raise SystemExit(exit_code)


def keyword_search(query: str, limit: int) -> list[dict[str, Any]]:
    from lib.keyword_search import InvertedIndex

    idx = InvertedIndex()
    idx.load()
    rows = []
    for rank, (movie, score) in enumerate(idx.bm25_search(query, limit), start=1):
        rows.append(
            normalize_result(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie.get("description", ""),
                    "score": score,
                },
                rank,
            )
        )
    return rows


def semantic_search(query: str, limit: int) -> list[dict[str, Any]]:
    from lib.semantic_search import MODEL_NAME, SemanticSearch, cosine_similarity

    searcher = SemanticSearch(MODEL_NAME)
    movies = load_movies()
    searcher.load_embeddings(movies)
    query_embedding = searcher.generate_embedding(query)
    scored: list[dict[str, Any]] = []
    for doc_embed, doc in zip(searcher.embeddings, searcher.documents):
        score = cosine_similarity(query_embedding, doc_embed)
        scored.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc.get("description", ""),
                "score": float(score),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [normalize_result(item, rank) for rank, item in enumerate(scored[:limit], start=1)]


def weighted_search(query: str, limit: int, alpha: float) -> list[dict[str, Any]]:
    from lib.hybrid_search import HybridSearch, hybrid_score

    hs = HybridSearch(load_movies())
    combined = hs.weighted_search(query, alpha, limit)
    for row in combined:
        row["hybrid_score"] = hybrid_score(row["bm25_score"], row["sem_score"], alpha)
        row["score"] = row["hybrid_score"]
    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return [normalize_result(item, rank) for rank, item in enumerate(combined[:limit], start=1)]


def hybrid_search(
    query: str,
    limit: int,
    rerank: str | None,
    enhance: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    from lib.hybrid_search import _get_hybrid_search

    search_query = query
    if enhance:
        from test_llm import llm_query

        search_query = llm_query(query, enhance)

    hybrid = _get_hybrid_search()
    fused = hybrid.rrf_search(search_query, RRF_K)
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    if rerank == "individual":
        from test_llm import llm_rerank_query

        retrieval = limit * SEARCH_MULTIPLIER
        candidates = ranked[:retrieval]
        for result in candidates:
            result["rerank_score"] = llm_rerank_query(search_query, result)
            result["score"] = result["rerank_score"]
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        ranked = candidates[:limit]
    elif rerank == "batch":
        from test_llm import llm_rerank_batch

        retrieval = limit * SEARCH_MULTIPLIER
        ranked = llm_rerank_batch(search_query, ranked[:retrieval], limit)
        for result in ranked:
            result["score"] = result.get("rrf_score", 0.0)
    elif rerank == "cross_encoder":
        from test_llm import cross_encoder_func

        retrieval = limit * SEARCH_MULTIPLIER
        ranked = cross_encoder_func(search_query, ranked[:retrieval], limit)
        for result in ranked:
            result["score"] = result.get("cross_encoder_score", 0.0)
    else:
        ranked = ranked[:limit]
        for result in ranked:
            result["score"] = result.get("rrf_score", 0.0)

    return search_query, [
        normalize_result(item, rank) for rank, item in enumerate(ranked, start=1)
    ]


def rag_mode(mode: str, query: str, limit: int, enhance: str | None, rerank: str | None) -> tuple[str, str, list[dict[str, Any]]]:
    from lib.hybrid_search import rrf_search
    from test_llm import llm_citations, llm_questions, llm_summarize, rag_llm

    search_query = query
    if enhance:
        from test_llm import llm_query

        search_query = llm_query(query, enhance)

    retrieval_limit = limit
    if rerank:
        retrieval_limit = limit * SEARCH_MULTIPLIER
    results = rrf_search(search_query, RRF_K, retrieval_limit, rerank_method=rerank or None)

    if rerank == "individual":
        from test_llm import llm_rerank_query

        for result in results:
            result["rerank_score"] = llm_rerank_query(search_query, result)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        results = results[:limit]
    elif rerank == "batch":
        from test_llm import llm_rerank_batch

        results = llm_rerank_batch(search_query, results, limit)
    elif rerank == "cross_encoder":
        from test_llm import cross_encoder_func

        results = cross_encoder_func(search_query, results, limit)
    else:
        results = results[:limit]

    if mode == "rag":
        answer = rag_llm(search_query, results)
    elif mode == "summarize":
        answer = llm_summarize(search_query, results)
    elif mode == "citation":
        answer = llm_citations(search_query, results)
    else:
        answer = llm_questions(search_query, results)

    normalized = [normalize_result(item, rank) for rank, item in enumerate(results, start=1)]
    return search_query, answer, normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="WebFlix TUI JSON search adapter")
    parser.add_argument("--json", action="store_true", help="Emit JSON (required)")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--rerank", default="")
    parser.add_argument("--enhance", default="")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    if not args.json:
        emit({"ok": False, "error": "--json is required"}, 1)

    mode = args.mode.strip().lower()
    query = args.query.strip()
    limit = args.limit if args.limit and args.limit > 0 else DEFAULT_SEARCH_LIMIT
    rerank = args.rerank.strip() or None
    enhance = args.enhance.strip() or None

    if not query:
        emit({"ok": False, "error": "query is empty"}, 1)

    known = {"keyword", "semantic", "weighted", "hybrid", "rag", "summarize", "citation", "question"}
    if mode not in known:
        emit({"ok": False, "error": f"unknown mode: {mode}"}, 1)

    try:
        with stdout_to_stderr():
            answer = ""
            enhanced_query = query
            if mode == "keyword":
                results = keyword_search(query, limit)
            elif mode == "semantic":
                results = semantic_search(query, limit)
            elif mode == "weighted":
                results = weighted_search(query, limit, args.alpha)
            elif mode == "hybrid":
                enhanced_query, results = hybrid_search(query, limit, rerank, enhance)
            else:
                enhanced_query, answer, results = rag_mode(mode, query, limit, enhance, rerank)

        payload: dict[str, Any] = {
            "ok": True,
            "mode": mode,
            "query": query,
            "results": results,
        }
        if enhanced_query != query:
            payload["enhanced_query"] = enhanced_query
        if answer:
            payload["answer"] = answer
        emit(payload)
    except FileNotFoundError:
        emit(
            {
                "ok": False,
                "error": "Search index not found. Run: uv run python cli/keyword_search_cli.py build",
            },
            1,
        )
    except Exception as exc:
        emit({"ok": False, "error": str(exc)}, 1)


if __name__ == "__main__":
    main()
