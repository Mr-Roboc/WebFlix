A search engine for streaming platforms built with Python and RAG.

## Terminal UI (Go)

Interactive Gemini-style search bar. Existing argparse CLIs are unchanged; the TUI calls `cli/tui_search.py` for JSON results.

From the repo root:

```bash
cd tui
go run ./cmd/webflix
```

Or:

```bash
go -C tui run ./cmd/webflix
```

Type a movie query and press Enter. Type `/` for a command palette.

| Command | What it does |
| --- | --- |
| `/keyword` | BM25 keyword search |
| `/semantic` | Embedding search |
| `/hybrid` | RRF hybrid search (default) |
| `/weighted` | Alpha-weighted hybrid search |
| `/rag` `/summarize` `/citation` `/question` | RAG answer modes |
| `/rerank individual\|batch\|cross_encoder\|none` | Rerank hybrid/RAG hits |
| `/enhance spell\|rewrite\|expand\|none` | Query rewrite |
| `/limit N` | Result count |
| `/alpha 0.5` | Weight for `/weighted` |
| `/help` `/clear` `/quit` | Utility |

Default mode is `/hybrid`. LLM modes need `OPENROUTER_API_KEY`. BM25 search needs an index:

```bash
uv run python cli/keyword_search_cli.py build
```
