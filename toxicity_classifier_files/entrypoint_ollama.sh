#!/bin/bash
/bin/ollama serve &
pid=$!

echo "🔴 Waiting for Ollama to start..."
sleep 5

# Sprawdzamy czy model jest, jak nie to pobieramy
if ! ollama list | grep -q "llama3.2"; then
    echo "⬇️ Downloading Llama 3.2..."
    ollama pull llama3.2
else
    echo "🟢 Llama 3.2 already exists."
fi

wait $pid