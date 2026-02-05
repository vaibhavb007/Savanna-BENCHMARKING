#!/bin/bash
QUESTIONS="data/all_questions.json"
ANSWERS="data/all_answers.json"
IMAGES="Images_LR"

# List of models
MODELS=("llava-llama3:latest" "qwen2.5vl:7b")
# "llava:7b", "gemma3:4b" , "qwen2.5vl:3b"

for MODEL in "${MODELS[@]}"; do
    # replace ":" and "/" with "_" for safe filenames
    SAFE_MODEL=$(echo $MODEL | tr ':/' '_')
    OUTPUT="results/benchmark_results_${SAFE_MODEL}.csv"

    # clear old results
    rm -f $OUTPUT

    echo "=== Benchmarking $MODEL ==="

    for ((i=0; i<1000; i++)); do
        echo ">>> Running question $i / 1000"
        python3 llm_accuracy_benchmarking.py \
            --questions $QUESTIONS \
            --answers $ANSWERS \
            --image-dir $IMAGES \
            --model $MODEL \
            --output $OUTPUT \
            --index $i
    done
done
