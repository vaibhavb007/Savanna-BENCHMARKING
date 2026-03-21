#!/bin/bash
set -e

QUESTIONS="yesno_questions.json"
ANSWERS="yesno_answers.json"
IMAGES="Images_LR"

# Models to benchmark
MODELS=("gemma3:4b" "gemma3n:e2b" "qwen2.5vl:3b" "qwen2.5vl:7b")

# Define power modes (adjust these numbers based on `sudo nvpmodel -q`)
declare -A POWER_MODES=(
  #["15W"]=0
  #["25W"]=1
 #["MAXN SUPER"]=2 #do this one separately bc it will need a reboot
  ["7W"]=3 
)

# Number of questions to run
TOTAL=$(jq ".questions | length" $QUESTIONS)
LIMIT=100   # or use "$TOTAL" if you want all

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL=$(echo $MODEL | tr ':/' '_')

  for MODE_NAME in "${!POWER_MODES[@]}"; do
    MODE_ID=${POWER_MODES[$MODE_NAME]}
    echo "============================================================"
    echo ">>> Switching to Jetson power mode: $MODE_NAME (ID=$MODE_ID), Model: $MODEL"
    echo "============================================================"
    sudo nvpmodel -m $MODE_ID
    sudo jetson_clocks
    sleep 1

    OUTPUT="benchmark_results_YESNO_${SAFE_MODEL}_${MODE_NAME}.csv"
    rm -f "$OUTPUT"

    echo "=== Benchmarking $MODEL under $MODE_NAME on $LIMIT questions ==="
    for ((i=0; i<$LIMIT; i++)); do
        echo ">>> [${MODE_NAME}] Running question $i / $LIMIT"
        python3 llm_accuracy_benchmarking.py \
            --questions "$QUESTIONS" \
            --answers "$ANSWERS" \
            --image-dir "$IMAGES" \
            --model "$MODEL" \
            --output "$OUTPUT" \
            --index "$i"
    done
  done
done

echo "All power modes completed."

