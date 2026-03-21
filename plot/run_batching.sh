#!/usr/bin/env bash
MODEL="llava-llama3:latest"
QUESTIONS="yesno_questions.json"
ANSWERS="yesno_answers.json"
IMAGE_DIR="Images_LR"
OUTPUT_BASE="batch_benchmarks"
PORT_BASE=11400
BATCH_SIZES=(1 2)
MEM_SAFE_MARGIN_MB=2000
SLEEP_BETWEEN=10

mkdir -p "$OUTPUT_BASE"

get_mem_mb() {
  # Read system memory in MB using "free -m"
  local line
  line=$(free -m | awk '/^Mem:/ {print $3, $2}')
  # returns "<used> <total>"
  echo "$line"
}

print_mem() {
  read used total <<<"$(get_mem_mb)"
  local free=$((total - used))
  echo "[Memory] Used: ${used} MB / Total: ${total} MB → Free: ${free} MB"
}


wait_until_ready() {
  local port=$1
  for t in {1..15}; do
    if curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1; then
      echo "🟢 Server on port ${port} is ready"
      return 0
    fi
    sleep 2
  done
  echo "❌ Server on port ${port} never responded"
  return 1
}

for bs in "${BATCH_SIZES[@]}"; do
  echo ""
  echo "============================"
  echo "  Batch size: ${bs}"
  echo "============================"
  print_mem

  # launch servers
  SERVERS_PIDS=()
  for i in $(seq 0 $((bs-1))); do
    PORT=$((PORT_BASE + i))
    read used total <<<"$(get_mem_mb)"
    free=$((total - used))
    if (( free < MEM_SAFE_MARGIN_MB )); then
      echo "⚠️  Skipping new server, only ${free} MB free"
      break
    fi
    echo "🟢 Starting server on port ${PORT}"
    OLLAMA_HOST="http://127.0.0.1:${PORT}" ollama serve --port "${PORT}" >"server_${PORT}.log" 2>&1 &
    pid=$!
    SERVERS_PIDS+=("$pid")
    wait_until_ready "${PORT}" || { kill "$pid"; unset SERVERS_PIDS; break; }
  done

  if [ "${#SERVERS_PIDS[@]}" -eq 0 ]; then
    echo "⚠️  Could not start servers for batch ${bs}"
    continue
  fi

  echo "🚀 Running ${#SERVERS_PIDS[@]} concurrent benchmarks..."
  START=$(date +%s.%N)
  for i in $(seq 0 $((bs-1))); do
    PORT=$((PORT_BASE + i))
    OLLAMA_HOST="http://127.0.0.1:${PORT}" \
      python3 llm_accuracy_benchmarking.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --answers "$ANSWERS" \
        --image-dir "$IMAGE_DIR" \
        --index "$i" \
        --output "${OUTPUT_BASE}/b${bs}_i${i}.csv" &
  done
  wait
  END=$(date +%s.%N)
  DURATION=$(echo "$END - $START" | bc)
  echo "✅ Batch ${bs} finished in ${DURATION}s"

  for pid in "${SERVERS_PIDS[@]}"; do kill "$pid"; done
  sleep "$SLEEP_BETWEEN"
done

