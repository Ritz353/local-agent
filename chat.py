import ollama

# Connect to the Ollama server running on this PC
client = ollama.Client(host="http://127.0.0.1:11434")

response = client.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "In one sentence, what is a GPU?"}],
)

print(response.message.content)