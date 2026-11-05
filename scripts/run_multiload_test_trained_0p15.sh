#!/usr/bin/env bash
# Test one learning-method family trained at lambda=0.15 across frozen loads.
# Usage: bash scripts/run_multiload_test_trained_0p15.sh <method>
set -euo pipefail

METHOD="${1:?Usage: bash scripts/run_multiload_test_trained_0p15.sh <method>}"

PROJECT_ROOT="${JNCA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_ROOT="${JNCA_LAMBDA_0P15_TRAIN_ROOT:-$PROJECT_ROOT/runs_train_test_lambda_0p15_20261103/training}"
BENCH_ROOT="${JNCA_MULTILOAD_BENCH_ROOT:-$PROJECT_ROOT/runs_arrival_rate_0p1_to_1p0_20261030/benchmarks}"
RESULT_ROOT="${JNCA_MULTILOAD_TEST_ROOT:-$PROJECT_ROOT/runs_multiload_test_models_trained_at_0p15_20261105}"
GPU_LIST_TEXT="${JNCA_TEST_GPUS:-2 3 4 5 6}"
SEED_GPU_LIST_TEXT="${JNCA_SEED_GPUS:-}"
RATE_LIST_TEXT="${JNCA_RATES:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

read -r -a GPUS <<< "${GPU_LIST_TEXT}"
read -r -a RATES <<< "${RATE_LIST_TEXT}"
TRAIN_SEEDS=(1 2 3 4 5)
TEST_SEEDS=(20001 20002 20003 20004 20005)

if [[ -n "${SEED_GPU_LIST_TEXT}" ]]; then
    read -r -a SEED_GPUS <<< "${SEED_GPU_LIST_TEXT}"
    if [[ ${#SEED_GPUS[@]} -ne ${#TRAIN_SEEDS[@]} ]]; then
        echo "JNCA_SEED_GPUS must contain exactly five GPU ids, one for each training seed." >&2
        exit 1
    fi
else
    if [[ ${#GPUS[@]} -lt 1 ]]; then
        echo "JNCA_TEST_GPUS must contain at least one GPU id." >&2
        exit 1
    fi
fi

case "${METHOD}" in
    egat_no_truststate)
        SOLVER="gat_seq2seq_ppo"
        METHOD_FLAGS=(
            --normalize_ts true
            --incremental_security_shaping true
            --security_reward_weight 0.5
            --reward_failure_penalty 1.0
            --include_trust_in_state false
        )
        ;;
    egat_truststate)
        SOLVER="gat_seq2seq_ppo"
        METHOD_FLAGS=(
            --normalize_ts true
            --incremental_security_shaping true
            --security_reward_weight 0.5
            --reward_failure_penalty 1.0
            --include_trust_in_state true
        )
        ;;
    a2c_gcn)
        SOLVER="a2c_gcn_seq2seq"
        METHOD_FLAGS=(
            --normalize_ts false
            --incremental_security_shaping false
            --include_trust_in_state false
        )
        ;;
    two_stage_gcn_drl_adapted)
        SOLVER="two_stage_gcn_a2c_adapted"
        METHOD_FLAGS=(
            --normalize_ts false
            --incremental_security_shaping false
            --include_trust_in_state false
        )
        ;;
    *)
        echo "Unknown method: ${METHOD}" >&2
        exit 1
        ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT_ROOT}/logs"

run_seed() {
    local index="$1"
    local gpu="$2"
    local train_seed="${TRAIN_SEEDS[$index]}"
    local train_dir="${TRAIN_ROOT}/${METHOD}/train_seed_${train_seed}"
    local checkpoint
    checkpoint="$(find "${train_dir}" -name best_model.pkl -type f -print -quit)"

    if [[ -z "${checkpoint}" ]]; then
        echo "Missing best_model.pkl under: ${train_dir}" >&2
        return 1
    fi

    for rate in "${RATES[@]}"; do
        local rate_tag="${rate/./p}"
        local test_root="${BENCH_ROOT}/rate_${rate}/test"

        if [[ ! -d "${test_root}/seed_20001" ]]; then
            echo "Missing frozen test benchmark: ${test_root}" >&2
            return 1
        fi

        for test_seed in "${TEST_SEEDS[@]}"; do
            local output_dir="${RESULT_ROOT}/${METHOD}/train_seed_${train_seed}/arrival_rate_${rate_tag}/test_seed_${test_seed}"
            mkdir -p "${output_dir}"

            echo "[$(date '+%F %T')] START method=${METHOD} train=${train_seed} rate=${rate} test=${test_seed} gpu=${gpu}"
            CUDA_VISIBLE_DEVICES="${gpu}" python -u main.py \
                --solver_name "${SOLVER}" \
                --seed "${train_seed}" \
                --use_cuda true \
                --cuda_id 0 \
                --pretrained_model_path "${checkpoint}" \
                --num_train_epochs 0 \
                --num_epochs 1 \
                --train_only false \
                --p_net_dataset_path benchmark_jnca_v1/p_net \
                --v_nets_dataset_path "${test_root}/seed_${test_seed}" \
                --regenerate_v_nets false \
                --shortest_method bfs_shortest \
                --k_shortest 1 \
                --if_save_records true \
                --if_temp_save_records false \
                --run_dir "${output_dir}" \
                "${METHOD_FLAGS[@]}" \
                > "${output_dir}/run.log" 2>&1

            test -f "${output_dir}/global_summary.csv" || {
                echo "Missing summary: ${output_dir}" >&2
                tail -n 50 "${output_dir}/run.log" >&2
                return 1
            }
            grep -q "Loaded pretrained model" "${output_dir}/run.log" || {
                echo "Checkpoint was not confirmed as loaded: ${output_dir}" >&2
                return 1
            }
            echo "[$(date '+%F %T')] DONE method=${METHOD} train=${train_seed} rate=${rate} test=${test_seed}"
        done
    done
}

pids=()
if [[ -n "${SEED_GPU_LIST_TEXT}" ]]; then
    # Explicit seed-to-GPU mapping: positions map to training seeds 1..5.
    for index in "${!TRAIN_SEEDS[@]}"; do
        gpu="${SEED_GPUS[$index]}"
        run_seed "${index}" "${gpu}" > "${RESULT_ROOT}/logs/${METHOD}_train_seed_${TRAIN_SEEDS[$index]}_gpu_${gpu}.log" 2>&1 &
        pids+=("$!")
        echo "${METHOD}: train seed ${TRAIN_SEEDS[$index]}, GPU ${gpu}, PID $!"
    done
else
    gpu_count="${#GPUS[@]}"
    for worker_index in "${!GPUS[@]}"; do
        gpu="${GPUS[$worker_index]}"
        (
            # One worker owns one GPU.  With fewer than five GPUs, it processes
            # its assigned training seeds sequentially rather than oversubscribing.
            for ((index=worker_index; index<${#TRAIN_SEEDS[@]}; index+=gpu_count)); do
                run_seed "${index}" "${gpu}"
            done
        ) > "${RESULT_ROOT}/logs/${METHOD}_gpu_${gpu}.log" 2>&1 &
        pids+=("$!")
        echo "${METHOD}: GPU ${gpu}, PID $!"
    done
fi

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

if [[ ${status} -ne 0 ]]; then
    echo "${METHOD} multi-load test failed; inspect ${RESULT_ROOT}/logs." >&2
    exit ${status}
fi

echo "${METHOD} multi-load test completed under ${RESULT_ROOT}/${METHOD}"
