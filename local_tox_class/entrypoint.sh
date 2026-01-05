#!/bin/bash

# 1. Uruchom serwer Ollama w tle (&)
/bin/ollama serve &

# Zapisz ID procesu serwera (PID), żebyśmy mogli na niego czekać na końcu
pid=$!

# 2. Poczekaj chwilę, aż serwer wstanie (np. 5 sekund)
sleep 5

# 3. Sprawdź i pobierz model
echo "🔴 Checking for llama3.2 model..."
ollama pull llama3.2
echo "🟢 Model llama3.2 is ready!"

# 4. Czekaj na proces serwera (to utrzymuje kontener przy życiu)
wait $pid