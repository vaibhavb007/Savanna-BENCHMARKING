# Benchmark results for Qwen2.5-VL-3B
The repo contains benchmarking results for Qwen-2.5-VL-3B, evaluating the effectiveness of prompt caching and payload ordering while grouping by images.

## Accuracy & Performance
- **accuracy_table.png**: Displays the overall accuracy for each configuration. Both image_first runs achieved a superior accuracy of 55.00%, outperforming the text_first configurations which scored 48.48% (cram: 4096) and 48.00% (cram: 0).

- **accuracy_by_type.png**: Breaks down the accuracy metrics by specific question categories (Attribute, Comparison, Counting, Equality, Location, Yes/No). The image_first advantage is maintained across most categories, with notable performance bumps in "Yes/No" and "Attribute" questions.

## Compute Time & Latency

- **latency_breakdown.jpg**: Tracks the raw latency for every prompt in the dataset. image_first configurations maintain a highly consistent and fast latency of approximately 0.2 seconds per query. In contrast, text_first configurations hover around 0.8 seconds per query.

- **prefill_decode_breakdown_Qwen2_5_VL_3B__cram__4096__image_first_.png**: Demonstrates why the image_first configuration is so fast. The prefill time stays remarkably low (around 100ms) because the model reuses the existing in-VRAM image tokens. The occasional prefill spikes (around 600ms) represent the first time a new image is loaded into the context window.

- **prefill_decode_breakdown_Qwen2_5_VL_3B__cram__0__text_first_.jpg**: Shows the compute penalty of text_first processing. Because the text changes every prompt, the VRAM slot cache is continually invalidated, forcing a full prefill evaluation of around 700ms for almost every single query.

- **prefill_decode_breakdown_Qwen2_5_VL_3B__cram__4096__text_first_.jpg**: Mirrors the poor prefill performance of the 0MB cache run, with prefill times sitting between 600ms and 700ms. Notably, this run aborted early (around prompt 550), likely due to hardware limits being breached.

## Power & Energy Consumption

- **power_gpu.jpg**: Illustrates the smoothed average GPU power draw during inference. The text_first runs heavily tax the GPU, consistently drawing over 5.0W. The image_first runs are much more lightweight, fluctuating between 3.5W and 4.25W.

- **energy_per_token.png**: A bar chart detailing energy consumption broken down by token type. Output token generation for text_first setups is incredibly energy-intensive (nearly 4.0 Joules per token). The image_first setups require less than 1.0 Joule per output token.

- **energy_cdf.png**: A Cumulative Distribution Function (CDF) showing total energy expenditure per query. The image_first runs are tightly clustered on the left, costing only 1 to 3 Joules per query. The text_first runs are shifted entirely to the right, demanding a massive 8 to 14 Joules per query.

## Memory Footprint

- **prompt_cache_stats.png**: Tracks the RAM-resident prompt cache size across the run. Because the text_first run constantly invalidates its active slots, it continuously dumps state into the system RAM, causing the footprint to grow linearly to nearly 2000MB before aborting. The image_first run stays at zero, proving that a RAM prompt cache is entirely unnecessary when VRAM slot reuse is optimized.

- **memory_utilization_Qwen2_5_VL_3B__cram__4096__text_first_.png**: Illustrates a severe memory accumulation issue when using the 4096MB cache with a text-first payload. As the active in-VRAM slots are continually invalidated, system RAM usage climbs steadily until free RAM is completely exhausted. Consequently, the board begins heavily paging to swap memory around prompt 400. This triggers a sharp spike in swap usage up to the 3000 MB safety threshold, causing the benchmarking script to correctly abort around prompt 550 to prevent a system lockup.