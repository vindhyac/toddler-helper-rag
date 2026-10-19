"""Toddler Helper: a small RAG chatbot (retrieve entries, then answer with Claude Haiku)."""
import json
import os
import re

import anthropic
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
CLAUDE_MODEL = "claude-haiku-5-5"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------- 1. Load the knowledge base ----------
KB_PATH = os.path.join(HERE, "data", "toddler_guidance_chunks.jsonl")
if not os.path.exists(KB_PATH):  # real KB is private; fall back to the sample
    KB_PATH = os.path.join(HERE, "data", "sample_chunks.jsonl")
with open(KB_PATH, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
by_id = {r["id"]: r for r in rows}


def clean(v):
    if v is None:
        return "not specified"
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


# Extra search phrases (used only for finding entries, never shown to the user)
EXTRA_PHRASES = {
    "S-052": "hits or slaps other kids at daycare, nursery, preschool or kindergarten",
    "X-05": "potty training, toilet training, sleep through the night, night waking, "
            "sleep training, nutrition, allergies, screen time hours, autism, ADHD, "
            "diagnosis, teenagers",
    "X-03": "emergency, not breathing, choking, bleeding, unconscious, injured, poisoned",
    "X-01": "autism, developmental delay, speech delay, worried about development, pediatrician",
}


def search_key(r):
    ask = clean(r["parent_might_ask"])
    if ask.startswith("("):
        ask = ""
    return f"{r['title']}\n{ask}\n{EXTRA_PHRASES.get(r['id'], '')}".strip()


# ---------- 2. Embed the search keys (takes a few seconds at startup) ----------
embedder = SentenceTransformer(EMBED_MODEL)
key_vecs = embedder.encode([search_key(r) for r in rows], normalize_embeddings=True)

EMERGENCY = re.compile(
    r"(not breathing|can'?t breathe|cannot breathe|struggling to breathe|trouble breathing|"
    r"choking|unconscious|unresponsive|seizure|bleeding|swallowed|poison|overdose|"
    r"drown|head injury|blue lips)", re.I)
CRISIS = re.compile(
    r"((hurt|hit|shake|shook|harm)\w*\s+(my\s+)?(child|baby|toddler|son|daughter|kid)\b|"
    r"about to lose it|lose (it|my temper|control)|breaking point|can'?t cope)", re.I)
CITATION = re.compile(r"\s*\[[^\]]*\d{2,3}[^\]]*\]")

EMERGENCY_MSG = ("This sounds like it could be a medical emergency. Please stop here and call "
                 "your local emergency number now (112 in Germany and across the EU; 911 in the US). "
                 "If your child is not breathing, is choking, or is unresponsive, follow the "
                 "dispatcher's instructions. This tool is for everyday parenting questions, "
                 "not medical emergencies.")
REFUSAL = ("I am sorry, I cannot help you with this. "
           "Please reach out to an expert in the field to answer that for you.")

SYSTEM = f"""You are a warm, practical assistant helping a parent of a toddler.
You answer ONLY using the entries inside <knowledge>. Never use the word "books" or mention
"sources" or "entries" in your reply; just speak naturally, as a knowledgeable helper would.

Rules:
1. Base every piece of advice on the entries. Do not add outside advice, studies, or statistics.
2. After each piece of advice, cite the entry ID in square brackets, like [S-011].
3. If the entries do not address the parent's actual situation, or the topic is out of scope
   (entry X-05 lists what is out of scope), reply with exactly this and nothing else:
   "{REFUSAL}"
   Do not add anything before or after it, and do not name specific specialists.
4. If entries disagree (type D), explain both views briefly instead of picking silently.
5. Never diagnose a child or give medical advice. If the parent describes danger, injury, or
   feeling out of control, put the safety guidance (X-02, X-03) first and encourage real-world help.
6. Be kind and brief: one short line of empathy, then 3 to 5 concrete steps, then what to avoid.
   No lecturing.
7. Every sentence of advice must be traceable to an entry. Do not add your own reminders,
   encouragements, or tips, even sensible ones.
8. If you use an entry marked APP, say it is general safety guidance rather than researched advice."""


# ---------- 3. Retrieval ----------
def entry_block(r):
    return (f"[{r['id']}] {r['title']}\nSources: {clean(r['sources'])}\n"
            f"{r['title']}\n{r['text']}")


def see_also(entry_id):
    m = re.search(r"\*\*See also:\*\*\s*(.+)", by_id[entry_id]["text"])
    return re.findall(r"[PSRQDX]-\d{2,3}", m.group(1)) if m else []


def retrieve_ids(question, k=4):
    q = embedder.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]
    scores = key_vecs @ q
    ids = [rows[i]["id"] for i in np.argsort(-scores)[:k]]

    front = []
    if EMERGENCY.search(question):
        front.append("X-03")
    if CRISIS.search(question):
        front.append("X-02")
    ids = [i for i in front if i not in ids] + ids

    extras = []
    for eid in ids[:2]:                      # pull in the cross-referenced entries
        for sid in see_also(eid):
            if sid not in ids and sid not in extras and len(extras) < 2:
                extras.append(sid)
    ids += extras
    for must in ("X-05", "X-01"):            # always show the scope + referral entries
        if must not in ids:
            ids.append(must)
    return ids


# ---------- 4. Generation ----------
_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def last_user_message(history):
    for turn in reversed(history or []):
        if isinstance(turn, dict):
            if turn.get("role") == "user" and isinstance(turn.get("content"), str):
                return turn["content"]
        elif isinstance(turn, (list, tuple)) and turn and isinstance(turn[0], str):
            return turn[0]
    return None


def answer(message, history, show_citations=True):
    try:
        if EMERGENCY.search(message):
            return EMERGENCY_MSG
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "The app is missing its API key. Add ANTHROPIC_API_KEY in the Space settings."

        previous = last_user_message(history)
        search_text = f"{previous} {message}" if previous and len(message.split()) < 8 else message
        context = "\n\n---\n\n".join(entry_block(by_id[i]) for i in retrieve_ids(search_text))

        content = f"<knowledge>\n{context}\n</knowledge>\n\n"
        if previous and search_text != message:
            content += f"Earlier message from the parent: {previous}\n"
        content += f"Parent's question: {message}"

        msg = get_client().messages.create(
            model=CLAUDE_MODEL, max_tokens=2000, system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text if show_citations else CITATION.sub("", text)
    except Exception as e:  # keep the chat alive on any failure
        print("ERROR:", type(e).__name__, e)
        return f"Something went wrong on my side ({type(e).__name__}). Please try again."


# ---------- 5. The web page ----------
demo = gr.ChatInterface(
    fn=lambda message, history, show_cit: answer(message, history, show_cit),
    additional_inputs=[gr.Checkbox(value=True, label="Show citations (entry IDs)")],
    title="Toddler Helper",
    description=("Describe a toddler situation and get practical, research-based guidance. "
                 "This is general parenting information, not medical advice."),
    examples=[["She screams in the supermarket and everyone is staring", True],
              ["He hits me when he is angry", True],
              ["Bedtime takes two hours", True]],
)

if __name__ == "__main__":
    demo.launch()
