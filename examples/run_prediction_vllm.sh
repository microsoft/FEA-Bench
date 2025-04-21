export MAX_SEQ_LEN=128000
export MAX_GEN_LEN=4096

DATASET_PATH=feabench-data/FEA-Bench-v1.0-Oracle
MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct
RESULTS_ROOT_DIR=scripts/experiments/results_full

PROMPT_MODE=natural-detailed
python -m feabench.run_prediction \
    --dataset_name_or_path $DATASET_PATH \
    --model_type vllm \
    --model_name_or_path $MODEL_NAME \
    --input_text $PROMPT_MODE \
    --output_dir $RESULTS_ROOT_DIR/$PROMPT_MODE