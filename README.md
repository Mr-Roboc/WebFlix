# WebFlix

WebFlix is a movie search engine built with Python, retrieval-augmented generation, and a Go terminal UI. It supports keyword, semantic, weighted, and reciprocal-rank-fusion search, along with conversational answers over retrieved movies.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Go 1.22+
- `OPENROUTER_API_KEY` for the RAG and LLM-powered modes

Install the Python dependencies from the repository root:

```bash
uv sync
```

## Prepare the search index

Build the BM25 index before using keyword or hybrid search:

```bash
uv run python cli/keyword_search_cli.py build
```

The search data is loaded from `data/movies.json`. Generated indexes and embeddings are stored in `cache/`.

## Run the terminal UI

From the repository root:

```bash
go -C tui run ./cmd/webflix
```

Or:

```bash
cd tui
go run ./cmd/webflix
```

Type a movie query and press Enter. Type `/` to open the command palette. Press `Ctrl+C` to quit.

### TUI commands

| Command | Description |
| --- | --- |
| `/keyword` | BM25 keyword search |
| `/semantic` | Embedding semantic search |
| `/hybrid` | RRF hybrid search (default) |
| `/weighted` | Alpha-weighted hybrid search |
| `/rag` | Retrieve movies and generate an answer |
| `/summarize` | Summarize retrieved movies |
| `/citation` | Answer with source citations |
| `/question` | Conversational Q&A over retrieved movies |
| `/rerank individual\|batch\|cross_encoder\|none` | Configure result reranking |
| `/enhance spell\|rewrite\|expand\|none` | Configure query enhancement |
| `/limit N` | Set the number of results |
| `/alpha 0.5` | Set the BM25/semantic weight in weighted mode |
| `/help` | Show all commands |
| `/clear` | Clear the transcript |
| `/quit` | Exit WebFlix |

The TUI calls `cli/tui_search.py` as a JSON adapter, keeping the existing Python search and RAG implementations as the source of truth.

## Disclaimer

WebFlix cannot guarantee results for every query. The movie dataset may not contain the requested movie, or its information may not be up to date.

## Run the Python CLIs

The original command-line interfaces remain available from the repository root:

```bash
uv run python cli/augmented_generation_cli.py question "Which movies are about friendship?"
uv run python cli/hybrid_search_cli.py rrf-search "space adventure" --limit 5
uv run python cli/keyword_search_cli.py search "time travel"
```

LLM-backed commands require an OpenRouter key:

```bash
export OPENROUTER_API_KEY="your-key"
```

## Project layout

```text
WebFlix/
├── cli/       Python search, reranking, and RAG commands
├── data/      Movie and evaluation datasets
├── cache/     Search indexes and embedding caches
└── tui/       Go Bubble Tea terminal interface
```

## Test the TUI package

```bash
go -C tui test ./...
```
