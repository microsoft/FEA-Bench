# FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation

This repository is the official implementation of the paper "FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation." It can be used for baseline evaluation using the prompts mentioned in the paper.

The repository includes several functionalities, primarily for obtaining the full dataset, running model inference aligned with the paper, and evaluating the results. The complete pipeline is as follows:

## 1. Environment Setup

You can create a new Python environment and install all dependencies using:
```bash
pip install -e .
```
If you plan to use VLLM inference, ensure that the installed libraries match your hardware.

## 2. Building the Full Evaluation Dataset

Due to licensing and company policies, we cannot release the full dataset. Our published version ([https://huggingface.co/datasets/microsoft/FEA-Bench](https://huggingface.co/datasets/microsoft/FEA-Bench)) only includes essential attributes, and the remaining content needs to be scraped from GitHub.

To construct the full FEA-Bench dataset and save it in the `feabench-data` folder, run the following command. Note that you need to replace `GITHUB_TOKEN` with your own GitHub token, which should have read-only access to public repositories:
```bash
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
```

## 3. Running Model Inference

Our repository only provides inference methods consistent with those in the paper. Agentless and other agent-based inferences can use the `FEA-Bench-v1.0-Lite-Standard` dataset constructed in the previous step, which is aligned with the format of SWE-Bench.

### Example of VLLM Inference:
```bash
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
```

### Example of OpenAI API-style Inference:
(DEEPSEEK_TOKENIZER is only required when using DeepSeek model inference)
```bash
export DEEPSEEK_TOKENIZER_PATH="xxx"
export OPENAI_API_KEY="xxx"
export OPENAI_BASE_URL="https://api.deepseek.com"

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
```

After running the inference, you should see the output `.jsonl` result files in the specified `output_dir`.

## 4. Running Model Evaluation

Our evaluation process is based on the code provided by SWE-Bench. We have provided a patch file `swe-bench.diff` to include the environment configurations for the task instances we are involved in.

Clone the SWE-Bench repository and apply the patch:
```bash
mkdir -p evaluator
cd evaluator
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
git checkout a0536ee6f9fd5ff88acf17a36a384bf3da3d93d6
git apply ../../swe-bench.diff
conda create --name fea-eval python=3.11
conda activate fea-eval
pip install -e .
```

To verify that the FEA-Bench task instances can run correctly on your machine, you can build a gold result based on the dataset:
```bash
python -m feabench.get_gold_results \
    --dataset_name_or_path feabench-data/FEA-Bench-v1.0-Standard \
    --save_dir feabench-data/experiments/gold \
    --file_name Gold__FEABench_v1.0__test.jsonl
```

The command to run the evaluation script is as follows (using the gold result constructed above as an example):
```bash
python -m swebench.harness.run_evaluation \
    --dataset_name ../../feabench-data/FEA-Bench-v1.0-Standard \
    --predictions_path ../../feabench-data/experiments/gold/Gold__FEABench_v1.0__test.jsonl \
    --max_workers 10 \
    --cache_level instance \
    --timeout 900 \
    --run_id FEABench_v1_Gold
```
The usage is identical to SWE-Bench. You can set the cache level `cache_level` based on your disk size. You should then obtain a result file similar to the following `.json` format:
```json
{
    "total_instances": 1401,
    "submitted_instances": 1401,
    "completed_instances": 1401,
    "resolved_instances": 1401,
    "unresolved_instances": 0,
    "empty_patch_instances": 0,
    "error_instances": 0,
    ...
}
```

Congratulations! You have completed the usage of FEA-Bench. If you have any questions, please raise them in the issues.

---

For more details, please refer to the [FEA-Bench Paper](https://arxiv.org/abs/2503.06680).
If you find our work helpful, we would be grateful if you could cite our work.
```
@misc{li2025feabenchbenchmarkevaluatingrepositorylevel,
      title={FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation}, 
      author={Wei Li and Xin Zhang and Zhongxin Guo and Shaoguang Mao and Wen Luo and Guangyue Peng and Yangyu Huang and Houfeng Wang and Scarlett Li},
      year={2025},
      eprint={2503.06680},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2503.06680}, 
}
```



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
