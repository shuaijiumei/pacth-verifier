
export BYTED_RAY_SERVE_RUN_HOST="::"
ray stop --force
ray start --head \
  --node-ip-address=[@] \
  --dashboard-host=127.0.0.1 \
  --disable-usage-stats
python verl_utils/reward/model_server.py
# serve run verl_utils.reward.model_server:app_handle
# serve shutdown # for exit app