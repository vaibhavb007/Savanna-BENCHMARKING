#!/bin/bash

# Configuration
QUESTIONS_FILE="data/all_questions.json"
ANSWERS_FILE="data/all_answers.json"
IMAGE_DIR="Images_LR"
MODEL_NAME="Qwen2.5-VL-3B-Instruct-Q4_K_M"
ENGINE="llamacpp"
TOTAL_QUESTIONS=1000
OUTPUT_CSV="results_${MODEL_NAME}.csv"

echo "Starting benchmark for $MODEL_NAME on $ENGINE..."

for ((i=0; i<$TOTAL_QUESTIONS; i++)); do
    echo "Running index: $i / $((TOTAL_QUESTIONS - 1))"

    python3 llama_benchmarking.py \
        --questions "$QUESTIONS_FILE" \
        --answers   "$ANSWERS_FILE" \
        --image-dir "$IMAGE_DIR" \
        --model     "$MODEL_NAME" \
        --index     $i \
        --engine    "$ENGINE" \
        --output    "$OUTPUT_CSV"

    # Small pause between prompts
    sleep 1
done

echo "Benchmark complete! Results saved to $OUTPUT_CSV"
