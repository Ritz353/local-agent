"""Semantic (meaning-based) search over an Obsidian vault using Ollama embeddings.

How it works:
  1. Every note is split into overlapping chunks of about 1,000 characters.
  2. Each chunk is turned into an embedding: a list of numbers that captures its meaning.
  3. A question is embedded the same way, and chunks are ranked by cosine similarity,
     so text with a similar *meaning* ranks high even if it shares no exact words.

Embeddings are saved to .notes_index.json, and only notes that changed since the last
run are re-embedded, so rebuilding the index is fast after the first time.
"""
import json
import math
import os
from pathlib import Path

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
INDEX_FILE = Path(__file__).with_name(".notes_index.json")
CHUNK_CHARS = 1000
OVERLAP = 150

# Run the small embedding model on the CPU (num_gpu=0) so it never competes with the chat
# model for GPU memory. Otherwise Ollama swaps models in and out of VRAM on every search.
EMBED_OPTIONS = {"num_gpu": 0} if os.environ.get("EMBED_ON_GPU") != "1" else {}

# nomic-embed-text was trained with these task prefixes; they improve retrieval quality.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at line breaks."""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)  # end on a line break if possible
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # overlap so ideas split across chunks aren't lost
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: 1.0 = same direction (same meaning), 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def build_index(client, notes, verbose: bool = True) -> dict:
    """Embed every note, reusing saved embeddings for notes that haven't changed."""
    old = {}
    if INDEX_FILE.exists():
        old = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    old_notes = old.get("notes", {}) if old.get("model") == EMBED_MODEL else {}

    notes_index = {}
    for path in notes:
        key, mtime = str(path), path.stat().st_mtime
        if key in old_notes and old_notes[key]["mtime"] == mtime:
            notes_index[key] = old_notes[key]  # unchanged: reuse
            continue
        chunks = chunk_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not chunks:
            continue
        # Put the note's title in front of every chunk so each one keeps its context.
        inputs = [f"{DOC_PREFIX}{path.stem}\n{c}" for c in chunks]
        vectors = client.embed(model=EMBED_MODEL, input=inputs, options=EMBED_OPTIONS).embeddings
        notes_index[key] = {"name": path.stem, "mtime": mtime, "chunks": chunks, "vectors": vectors}
        if verbose:
            print(f"  indexed {path.stem} ({len(chunks)} chunks)")

    index = {"model": EMBED_MODEL, "notes": notes_index}
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    return index


def semantic_hits(client, index: dict, query: str, top: int = 5) -> list[tuple[float, str, str]]:
    """Return (score, note name, best chunk) for the notes most similar in meaning to the query."""
    qvec = client.embed(model=EMBED_MODEL, input=f"{QUERY_PREFIX}{query}", options=EMBED_OPTIONS).embeddings[0]
    best = {}  # note name -> (score, chunk)
    for note in index["notes"].values():
        for chunk, vec in zip(note["chunks"], note["vectors"]):
            score = cosine(qvec, vec)
            if score > best.get(note["name"], (-1.0, ""))[0]:
                best[note["name"]] = (score, chunk)
    ranked = sorted(((s, name, c) for name, (s, c) in best.items()), reverse=True)
    return ranked[:top]
