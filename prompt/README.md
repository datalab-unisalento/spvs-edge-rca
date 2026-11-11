# Prompts and inference

system_prompt.txt and user_prompt.txt are extracted from the recorded Gemma prompt. Raw inputs and outputs for all 30 runs are in data/runs/; original campaign records are in data/episodes/. The input_record/output_record objects preserve LM Studio records without normalizing responses.

| Parameter | Gemma | Qwen |
|---|---|---|
| Logged identifier | google/gemma-4-e4b | qwen/qwen3.5-9b |
| Quantization | q4_k_m | q4_k_m |
| Runtime | LM Studio 0.4.12 | LM Studio 0.4.12 |
| temperature | 0.2 | 0.2 |
| Context window (tokens) | 4096 | 4096 |
| top_p | 0.9 | 0.95 |
