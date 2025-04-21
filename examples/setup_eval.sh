mkdir -p evaluator
cd evaluator
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
git checkout a0536ee6f9fd5ff88acf17a36a384bf3da3d93d6
git apply ../../swe-bench.diff
conda create --name fea-eval python=3.11
conda activate fea-eval
pip install -e .