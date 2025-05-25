## Installation

### Quickstart: SHuBERT inference 
**Important**: Make sure the `SBATCH` flags, `CONDA_ROOT`, `env_name`, `csv_path`, `checkpoint_path`, `output_dir` in `shubert_inference.sh` before running it. For downloading model weights, see [Model Weights](README.md#2-model-weights) section in the main README.
```bash
# In SHuBERT folder
conda env create -f environment_shubert.yml
conda activate shubert_train_env
cd fairseq
pip install -e .
cd ../features
bash shubert_inference.sh
```
