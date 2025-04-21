# extract gold patch as results
python -m feabench.get_gold_results \
    --dataset_name_or_path feabench-data/FEA-Bench-v1.0-Standard \
    --save_dir feabench-data/experiments/gold \
    --file_name Gold__FEABench_v1.0__test.jsonl



# evaluation directory
cd evaluator
cd SWE-bench

# evaluate
python -m swebench.harness.run_evaluation \
    --dataset_name ../../feabench-data/FEA-Bench-v1.0-Standard \
    --predictions_path ../../feabench-data/experiments/gold/Gold__FEABench_v1.0__test.jsonl \
    --max_workers 10 \
    --cache_level instance \
    --timeout 900 \
    --run_id FEABench_v1_Gold
