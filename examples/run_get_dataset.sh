export GITHUB_TOKEN="xxx"

python -m feabench.get_dataset \
    --dataset microsoft/FEA-Bench \
    --testbed feabench-data/testbed \
    --lite_ids instances_lite.json \
    --medium_file feabench-data/FEA-Bench-v1.0-medium.jsonl \
    --standard_dataset_path feabench-data/FEA-Bench-v1.0-Standard \
    --oracle_dataset_path feabench-data/FEA-Bench-v1.0-Oracle \
    --lite_standard_dataset_path feabench-data/FEA-Bench-v1.0-Lite-Standard \
    --lite_oracle_dataset_path feabench-data/FEA-Bench-v1.0-Lite-Oracle
