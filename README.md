# Toddler Helper: a RAG chatbot for toddler-parenting advice

Describe a toddler problem ("he hits when another child takes his toy") and get a short, step-by-step answer grounded in a curated knowledge base, with safety handling and a fixed refusal for anything out of scope.

I built this to learn Retrieval-Augmented Generation end to end, and then gave it a UI I'd be happy to ship.

## How it works

```
question -> safety regex (emergency / crisis) -> embed query -> top-k entries (Chroma / cosine)
         -> add "See also" + always-on safety entries -> Claude Haiku (rules prompt) -> formatted answer
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | One knowledge-base entry per chunk (108 chunks) | Entries are self-contained; fixed 128/512-token splits cut entries mid-advice |
| Search key | Title + "Parent might ask" phrases | Matches how parents phrase problems, not how books do |
| Embeddings | `BAAI/bge-small-en-v1.5` with the query prefix | Small, free, strong for retrieval |
| Store | Chroma, cosine similarity | Simple local vector DB |
| Generation | Claude Haiku with a rules prompt | Every advice sentence must trace to a retrieved entry |
| Safety | Code-level regexes, forced emergency/crisis entries, fixed refusal sentence | Similarity scores cannot detect out-of-scope questions |

## Evaluation

- 25-question golden set plus 8 held-out questions, scored with Recall@k and MRR.
- Key finding: out-of-scope questions still score about 0.6+ similarity, so refusal can't rely on a retrieval threshold. It is handled in the prompt, with always-included safety entries and regex checks.
- Failure analysis: one apparent content gap (S-052) was actually a retrieval miss, fixed by adding phrases to its search key.

(Add your final Recall@k / MRR table here from the Colab runs.)

## Two versions

1. **Python/Gradio app** (`app.py`): the full pipeline with embeddings and a vector search. Run locally or in Colab.
2. **Web demo** (`web/template.html`): a single-page UI built as a Claude artifact. It runs on the viewer's own Claude account, so there is no API key and no hosting cost to me. Retrieval is done by Claude choosing entries from an index of titles.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python app.py
```

Without the private knowledge base, the app falls back to `data/sample_chunks.jsonl` (5 original entries).

## Knowledge base

The full knowledge base is paraphrased from parenting books I own, so it is **not published** here. `data/sample_chunks.jsonl` shows the schema: `id, title, type, category, ages, sources, parent_might_ask, text`.

## Limitations

- Advice quality is bounded by the knowledge base; it is not medical or professional advice.
- The web demo's retrieval has not been scored against the golden set.

## Demo

(Add a short screen recording or GIF here.)
