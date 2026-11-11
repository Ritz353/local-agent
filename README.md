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

## Search

`notes_search.py` splits each note into overlapping ~1,000-character chunks, embeds them with `nomic-embed-text` through Ollama, and ranks chunks by cosine similarity to the question, so text with a similar *meaning* ranks high even when it shares no exact words. Embeddings are cached in `.notes_index.json` (gitignored), and only changed notes are re-embedded.

`SEARCH_MODE` picks the strategy: `keyword` (the original), `semantic`, or `hybrid` (default: the top meaning matches plus strong keyword matches, since exact terms like "Lecture 11" still matter).

## Tools

| Tool | What it does |
|---|---|
| `search_notes` | Hybrid search across an Obsidian vault: embedding similarity (meaning) plus keyword matches |
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
ollama pull nomic-embed-text
python agent.py
```

Set the `OBSIDIAN_VAULT` environment variable to point at your own vault.

## Tests and evaluation

```
pytest -v          # unit tests
python evals.py    # agent evaluation (needs Ollama running)
```

The unit tests build a small fake vault in a temporary folder, so they never touch real notes.

## Evaluation

`evals.py` runs a 35-question suite through the agent and scores **tool choice** (did it call the tools it needed, and none when none were needed) and **answer accuracy** (does the answer contain the known fact). Categories include exact math, general knowledge that should need no tools, questions about my notes, hard note questions (misspellings, no obvious keywords), multi-step questions, and **honesty checks**: questions my notes can't answer, where the agent must say so instead of inventing something.

Results with `qwen3.5:9b`, averaged over 3 runs per question (`python evals.py --repeat 3`):

| Category | Keyword search | Hybrid search |
|---|---|---|
| Time | 12/12 | 12/12 |
| Math | 18/18 | 18/18 |
| No-tool (general knowledge) | 18/18 | 18/18 |
| Notes | 27/27 | 27/27 |
| Notes – hard | 11/15 | **13/15** |
| Honesty | 6/6 | 6/6 |
| Multi-step | 9/9 | 9/9 |
| **Overall** | **96%** | **98%** |
| Avg time per question | 9.7 s | 12.2 s |

**How it got there:** the first run scored 86%. Reading the failures showed three causes:
1. The system prompt made the agent search my notes even for general questions, so I limited note searches to questions about *my* notes.
2. On questions with no answer in my notes, it kept searching until it hit the step limit, so I added a search budget and a final step that removes the tools and forces an answer.
3. The eval itself rejected correct "I couldn't find it" answers phrased in ways I hadn't listed, so I fixed the checker.

The prompt fix caused a regression (the agent started doing multiplication in its head), which the eval caught. An explicit rule to always use the math and time tools fixed it.

**Adding semantic search:** keyword search failed on questions like a misspelled "twos compliment" or "explain recursion using my class notes". Hybrid search, which combines embedding similarity with keyword matches, fixed both, raising notes-hard from 11/15 to 13/15. The cost is about 2.5 s more per question.

The remaining miss is a tool-choice issue, not a search issue: for "Based on what I've studied, how are negative numbers stored in binary?", the agent answers correctly from general knowledge without checking my notes. I left it rather than adding that exact phrase to the prompt, which would just overfit the eval.

**Performance note:** at first, 12% of the model spilled from the 8 GB GPU onto the CPU, which bottlenecked everything. Enabling flash attention and an 8-bit KV cache in Ollama (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) brought it to 100% GPU.

## Next steps

- Better tool choice for indirect questions about my notes
- Re-ranking search results with the LLM
- More tools, such as checking the status of my homelab services
