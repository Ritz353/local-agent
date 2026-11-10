import datetime
import os
from pathlib import Path

import ollama

client = ollama.Client(host="http://127.0.0.1:11434")
MODEL = "qwen3.5:9b"
VAULT = Path(os.environ.get("OBSIDIAN_VAULT", r"C:\Users\sebas\Documents\Obsidian\Work"))
SKIP_DIRS = {".venv", ".obsidian", ".trash", "__pycache__", "node_modules", ".git"}

SYSTEM_PROMPT = (
    "You are a helpful assistant for Wilfredo, a computer science student. "
    "You can search and read his Obsidian notes. When a question is about his classes, "
    "notes, projects, or anything he has written, first use search_notes, then use "
    "read_note on the most relevant result before answering. Always say which note(s) "
    "you used. If the notes don't contain the answer, say so instead of guessing."
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


def search_notes(query: str) -> str:
    """Search Wilfredo's Obsidian notes by keywords and list the best matching notes.

    Args:
        query: A few keywords describing what to look for, e.g. "two's complement"

    Returns:
        The names of the most relevant notes with a short snippet from each.
    """
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
    if not results:
        return "No notes matched those keywords."
    return "\n\n".join(f"NOTE: {name}\n...{snip}..." for _, name, snip in results[:5])


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

def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(6):  # safety limit
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"num_ctx": 8192},  # room for note text in the conversation
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
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