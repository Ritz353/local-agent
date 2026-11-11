import datetime
import os
from pathlib import Path

import ollama

import notes_search

client = ollama.Client(host="http://127.0.0.1:11434")
MODEL = os.environ.get("AGENT_MODEL", "qwen3.5:9b")   # override to compare models
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "8192"))
MAX_STEPS = 6  # most model calls allowed per question
VAULT = Path(os.environ.get("OBSIDIAN_VAULT", r"C:\Users\sebas\Documents\Obsidian\Work"))
SKIP_DIRS = {".venv", ".obsidian", ".trash", "__pycache__", "node_modules", ".git"}
SEARCH_MODE = os.environ.get("SEARCH_MODE", "hybrid")  # keyword | semantic | hybrid

SYSTEM_PROMPT = (
    "You are a helpful assistant for Wilfredo, a computer science student. "
    "You can search and read his Obsidian notes, but only do that when he asks about HIS "
    "notes, classes, projects, or something he wrote (for example 'my notes', 'my lecture', "
    "'what did I write'). For general knowledge questions, answer directly without tools. "
    "Exception: ALWAYS use the multiply tool for multiplication or percentages (never do the "
    "math in your head), and ALWAYS use get_current_time for anything about the current date or time. "
    "When you do use his notes: use search_notes, then read_note on the most relevant result, "
    "and say which note(s) you used. If two searches don't find anything relevant, stop "
    "searching and tell him his notes don't seem to cover it. Never guess or invent what "
    "his notes say."
)


# ---------- TOOLS ----------

def get_current_time() -> str:
    """Get the current local date and time.

    Returns:
        The current date and time as text.
    """
    return datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")


def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of a and b
    """
    return float(a) * float(b)


def all_notes():
    """Yield every Markdown note in the vault, skipping code/system folders."""
    for path in VAULT.rglob("*.md"):
        if not SKIP_DIRS.intersection(path.parts):
            yield path


def keyword_hits(query: str) -> list[tuple[float, str, str]]:
    """Rank notes by how often the query's words appear (title matches count extra)."""
    words = [w for w in query.lower().split() if len(w) > 2] or [query.lower()]
    results = []
    for path in all_notes():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        title = path.stem.lower()
        score = sum(lower.count(w) for w in words) + sum(10 for w in words if w in title)
        if score:
            hit = min((lower.find(w) for w in words if w in lower), default=0)
            snippet = " ".join(text[max(0, hit - 120): hit + 200].split())
            results.append((score, path.stem, snippet))
    results.sort(reverse=True)
    return results


def format_hits(hits) -> str:
    if not hits:
        return "No notes matched that search."
    return "\n\n".join(f"NOTE: {name}\n...{' '.join(snip.split())[:320]}..." for _, name, snip in hits)


def keyword_search(query: str) -> str:
    """Keyword-only search (the original version, kept for comparison and tests)."""
    hits = keyword_hits(query)
    return format_hits(hits[:5]) if hits else "No notes matched those keywords."


_index = None


def get_index():
    """Build (or refresh) the embedding index the first time it's needed."""
    global _index
    if _index is None:
        _index = notes_search.build_index(client, all_notes())
    return _index


def search_notes(query: str) -> str:
    """Search Wilfredo's Obsidian notes and list the best matching notes.

    Args:
        query: What to look for, in a few words, e.g. "two's complement"

    Returns:
        The names of the most relevant notes with a short snippet from each.
    """
    if SEARCH_MODE == "keyword":
        return keyword_search(query)
    semantic = notes_search.semantic_hits(client, get_index(), query, top=5)
    if SEARCH_MODE == "semantic":
        return format_hits(semantic)
    # hybrid: the top meaning-based matches, plus strong keyword matches they missed
    # (exact keywords still matter for things like "Lecture 11" or "Ch 2").
    merged = semantic[:4]
    names = {name for _, name, _ in merged}
    for hit in keyword_hits(query):
        if len(merged) >= 6:
            break
        if hit[1] not in names:
            merged.append(hit)
            names.add(hit[1])
    return format_hits(merged)


def read_note(name: str) -> str:
    """Read the full text of one of Wilfredo's notes.

    Args:
        name: The note's name exactly as shown by search_notes

    Returns:
        The note's content (long notes are cut off).
    """
    for path in all_notes():
        if path.stem.lower() == name.lower():
            return path.read_text(encoding="utf-8", errors="ignore")[:6000]
    return f"No note named '{name}'. Use search_notes to find the right name."


TOOLS = {
    "get_current_time": get_current_time,
    "multiply": multiply,
    "search_notes": search_notes,
    "read_note": read_note,
}


# ---------- THE AGENT LOOP ----------

def run_agent(question: str, trace: list | None = None, verbose: bool = True) -> str:
    """Answer a question, calling tools as needed.

    If a `trace` list is passed in, the name of every tool the model calls is
    appended to it (the evaluation script uses this to score tool choice).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(MAX_STEPS):  # safety limit
        last_step = step == MAX_STEPS - 1
        if last_step:
            # Out of tool calls: take the tools away and make it answer with what it has,
            # instead of quitting with no answer.
            messages.append({"role": "system", "content":
                             "You have used all your tool calls. Answer now using what you found, "
                             "or say clearly that you couldn't find it."})
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=None if last_step else list(TOOLS.values()),
            options={"num_ctx": NUM_CTX},  # room for note text in the conversation
        )
        messages.append(response.message)

        if last_step or not response.message.tool_calls:
            return response.message.content or "I couldn't come up with an answer."

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if trace is not None:
                trace.append(name)
            if verbose:
                print(f"  [tool] {name}({args})")
            try:
                result = TOOLS[name](**args)
            except Exception as e:
                result = f"Error: {e}"
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Stopped: too many steps."


if __name__ == "__main__":
    while True:
        question = input("\nYou: ")
        if question.lower() in ("quit", "exit"):
            break
        print("Agent:", run_agent(question))