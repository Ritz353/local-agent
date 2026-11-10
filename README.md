# Local AI Agent

A tool-calling AI agent that runs entirely on my own GPU (RTX 3070) using [Ollama](https://ollama.com) and Qwen 3.5.
It decides which tools to use, runs them through an agent loop, and answers from the results,
including searching and reading my Obsidian notes. Nothing is sent to a cloud API.

## How it works

1. The question, a system prompt, and descriptions of the available tools are sent to the model.
2. If the model requests a tool, my code runs the matching Python function and returns the result as a `tool` message.
3. This repeats, up to a safety limit, until the model gives a final answer.

The model never executes anything itself: it can only *request* a tool, and the code decides what actually runs.
Errors from a tool are returned to the model instead of crashing the program.

## Tools

| Tool | What it does |
|---|---|
| `search_notes` | Keyword search across an Obsidian vault, skipping code and system folders |
| `read_note` | Reads a note in full (truncated for long notes) |
| `get_current_time` | Current date and time, which an LLM can't know on its own |
| `multiply` | Exact arithmetic, which small LLMs often get wrong |

## Example

```
You: According to my notes, what is two's complement?
  [tool] search_notes({'query': "two's complement"})
  [tool] read_note({'name': 'Lecture 05 - Signed Numbers, Two's Complement and ASCII'})
Agent: Based on your Lecture 05 note, two's complement represents negative numbers by ...
```

## Run it

```
pip install -r requirements.txt
ollama pull qwen3.5:9b
python agent.py
```

Set the `OBSIDIAN_VAULT` environment variable to point at your own vault.

## Tests

```
pytest -v
```

The tests build a small fake vault in a temporary folder, so they never touch real notes.

## Next steps

- Semantic (embedding-based) search instead of keywords
- An evaluation set to measure how often the agent picks the right tool
- More tools, such as checking the status of my homelab services
