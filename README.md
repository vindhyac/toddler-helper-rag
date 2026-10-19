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

25-question golden set (each question lists the entry IDs a correct answer needs), scored with Recall@k and MRR.

| Metric | Baseline (title-only search key) |
|---|---|
| Recall@1 | 68% (17/25) |
| Recall@3 | 72% (18/25) |
| Recall@5 | 72% (18/25) |
| MRR | 0.70 |

**What the 7 misses at k=3 showed**

- **5 of 7 were safety or out-of-scope questions** (potty training, baby sleep, screen time, "might have autism", "struggling to breathe"). Their expected entries are refusal/emergency entries (`X-01`, `X-03`, `X-05`), which are written as instructions, not topics, so similarity search can't find them. Out-of-scope questions also score about 0.6+ similarity against real entries, so a score threshold can't detect them either.
- **Decision:** safety does not depend on retrieval. Emergency and crisis phrases are caught by regex before search, `X-05` and `X-01` are always included in the prompt, and the refusal sentence is fixed.
- **2 of 7 were real retrieval misses** ("he slaps other kids" -> hitting entry `S-052`; "I yelled at her and feel terrible" -> repair entries `R-01`, `R-05`). The `S-052` miss looked like a content gap but was a vocabulary gap, fixed by adding "Parent might ask" phrases to the search key.

(Add the improved search-key and held-out results here.)

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
