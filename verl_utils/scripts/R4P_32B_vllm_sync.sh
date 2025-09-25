
### PREPROCESS:
# cd /opt/tiger/patch_verifier
# pip install -e .

# sudo sed -i '352,354d' /usr/local/lib/python3.11/dist-packages/vllm/v1/engine/processor.py
# git config --global safe.directory "*"

# pip install pylint

export ROOT_DIR='/mnt/bn/-research-models/'
export BASE_MODEL=$ROOT_DIR'/models/Qwen3-32B'
export WAND_PROJECT='patch_verifier'
export EXPERIMENT_NAME='R4P_32B'

sleep 5m

git config --global safe.directory "*"
pip install pylint

train_files="['$ROOT_DIR/datasets/data_train_verb.parquet']"
test_files="['$ROOT_DIR/datasets/data_test_verb.parquet']"

mkdir /opt/tiger/patch_verifier/workspace
tool_root_path=/mnt/bn/-research-models//datasets/benchmarks
tool_temp_path=/opt/tiger/patch_verifier/workspace

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=loop \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-length-norm" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    +actor_rollout_ref.rollout.tool_root_path=$tool_root_path \
    +actor_rollout_ref.rollout.tool_temp_path=$tool_temp_path \
    +actor_rollout_ref.rollout.enable_write=false \
    +actor_rollout_ref.rollout.max_turns=50 \
    +actor_rollout_ref.rollout.generation_timeout=1800 \
    +actor_rollout_ref.rollout.max_single_turn_tokens=30760 \
    +actor_rollout_ref.rollout.enable_compact_filtering=true \
    +actor_rollout_ref.rollout.enable_qwen3_thinking_in_multiturn=true \
    +actor_rollout_ref.rollout.enable_turn_reminder=false \
    +data.enable_qwen3_thinking=true \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.shuffle=True \
    +data.seed=42 \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=30760 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm_with_tool \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    custom_reward_function.path=verl_utils/reward/reward_fn.py \
    custom_reward_function.name=compute_score_verb \
    reward_model.reward_manager=naive \
    2>&1 | tee $EXPERIMENT_NAME.log