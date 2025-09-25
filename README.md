## Patch Verifier

Codebase for the paper “Patch Verifier”. We use reinforcement learning to fine-tune Qwen to judge software patch correctness. Most runnable scripts live under `verl_utils/`.

### Key Directories (under `verl_utils/`)
- `data/`: Rollout and verification data pipelines (Parquet I/O, prompts, utils).
- `tool/`: Tool schemas and lightweight code-search/edit tools (`lite_tool.py`, configs in `tool/config/`).
- `reward/`: Reward server/client to score patch candidates (`model_server.py`, `model_client.py`).
- `eval/`: Result evaluation utilities (e.g., majority voting, selectors).
- `scripts/`: Convenience launch scripts for different model sizes.

### Install
```bash
pip install -r requirements.txt
pip install -e .
```

### Quickstart
1) Start reward server (optional, for remote scoring or RL):
- Edit `MODEL_PATH` in `verl_utils/reward/model_server.py` to your Qwen checkpoint.
- Then run:
```bash
ray stop --force
ray start --head --disable-usage-stats
python verl_utils/reward/model_server.py
```

2) Generate rollouts (no tool, naive):
```bash
python -m verl_utils.data.rollout \
  --model claude37 \
  --data_path data/data_test_batch_without_tool.parquet \
  --naive True
```

3) Judge/verify (produce labels) with lightweight tool config:
```bash
python -m verl_utils.data.ver \
  --model claude37 \
  --root data/datasets \
  --split test \
  --tool_config_path verl_utils/tool/config/lite_tool_config.yaml
```

Notes
- If you evaluate via an external harness, set `HARNESS_URL` and `SERVER_URL` in `verl_utils/reward/model_client.py`.
- Data expectations: `ver.py` looks for `{root}/data_[train|test]_ver.parquet`; `rollout.py` reads Parquet inputs you provide.

### Acknowledgements
Built on the VERL framework and Qwen models.
