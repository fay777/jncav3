#!/usr/bin/env bash
# Controlled EGAT-DPOT experiment: only the PPO security/efficiency reward
# mixing weight differs from the current formal configuration.
set -euo pipefail

RUN_ROOT="runs_formal_20261023_egat_fixedmix"
mkdir -p "${RUN_ROOT}/logs"

# Change these IDs only when the server scheduler assigns different GPUs.
SEEDS=(1 2 3 4 5)
GPUS=(2 3 4 5 6)

for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="${GPUS[$index]}"
    run_dir="${RUN_ROOT}/train_seed${seed}_gpu${gpu}"
    log_file="${RUN_ROOT}/logs/seed${seed}_gpu${gpu}.log"

    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -u main.py \
        --solver_name gat_seq2seq_ppo \
        --seed "${seed}" \
        --use_cuda true \
        --cuda_id 0 \
        --p_net_dataset_path benchmark_jnca_v1/p_net \
        --validation_v_nets_dataset_path benchmark_jnca_v1/validation/seed_10001 \
        --shortest_method bfs_shortest \
        --k_shortest 1 \
        --normalize_ts true \
        --incremental_security_shaping true \
        --security_reward_weight 0.5 \
        --reward_failure_penalty 1.0 \
        --num_train_epochs 140 \
        --validation_interval 10 \
        --train_only true \
        --run_dir "${run_dir}" \
        > "${log_file}" 2>&1 &

    echo "seed ${seed}: GPU ${gpu}, PID $!, log ${log_file}"
done
