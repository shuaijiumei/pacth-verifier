## Patch Verifier

Patch Verifier is the official codebase for the paper “Patch Verifier”. We use reinforcement learning to fine-tune Qwen models to judge software patch correctness from code context and diffs. The training and evaluation pipeline is built on the VERL framework.

### Highlights
- **RL for judging patches**: Train Qwen to assess whether a patch fixes an issue without regressions.
- **End-to-end pipeline**: Data preparation, training, and evaluation scripts/configs.
- **Scalable**: Supports single- and multi-GPU setups.

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Data
- Prepare your dataset of code contexts, patches (diffs), and correctness labels/scores.
- See `examples/data_preprocess/` for reference preprocessing scripts.

### Quickstart
- Configure training via VERL configs under `verl/trainer/config/` (modify as needed for Patch Verifier).
- Example (single GPU PPO trainer):
```bash
torchrun --nproc_per_node=1 -m verl.trainer.main_ppo \
  --config verl/trainer/config/ppo_trainer.yaml
```
- For evaluation:
```bash
python -m verl.trainer.main_eval \
  --config verl/trainer/config/evaluation.yaml
```

### Project Structure (partial)
- `verl/` — core training, models, workers, and utilities (VERL).
- `examples/` — data preprocessing and task examples.
- `recipe/` — runnable recipes and scripts for different training variants.


### Acknowledgements
Built on the VERL training framework and Qwen model family.


