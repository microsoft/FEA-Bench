export OPENAI_API_KEY="xxx"

DATASET_PATH=feabench-data/FEA-Bench-v1.0-Oracle
MODEL_NAME=deepseek-chat
RESULTS_ROOT_DIR=scripts/experiments/results_full

PROMPT_MODE=natural-detailed
python -m feabench.run_prediction \
    --dataset_name_or_path $DATASET_PATH \
    --model_type openai \
    --model_name_or_path $MODEL_NAME \
    --input_text $PROMPT_MODE \
    --output_dir $RESULTS_ROOT_DIR/$PROMPT_MODE \
    --num_proc 1