#!/usr/bin/env bash
# Train one EGAT variant at lambda=0.15 with explicit, verified parameters.
# Usage:
#   JNCA_SEED_GPUS="2 3 4 5 6" bash scripts/run_train_egat_lambda_0p15_variant.sh <variant>
# Variants: no_trust_fixed05, trust_fixed05, trust_ssi_adaptive
set -euo pipefail

VARIANT="${1:?Usage: bash scripts/run_train_egat_lambda_0p15_variant.sh <variant>}"
PROJECT_ROOT="${JNCA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_RUN_ROOT="${JNCA_LAMBDA_0P15_RUN_ROOT:-$PROJECT_ROOT/runs_train_test_lambda_0p15_20261104}"
VSETTING="${JNCA_LAMBDA_0P15_SETTING:-$PROJECT_ROOT/settings/v_sim_setting_lambda_0p15.yaml}"
P_NET="${JNCA_P_NET_DATASET:-$PROJECT_ROOT/benchmark_jnca_v1/p_net}"
VALIDATION_ROOT="${JNCA_LAMBDA_0P15_VALIDATION_ROOT:-$PROJECT_ROOT/benchmark_jnca_v1/validation/seed_10001_rate_0p15}"
SEED_GPU_LIST_TEXT="${JNCA_SEED_GPUS:-2 3 4 5 6}"

TRAIN_SEEDS=(1 2 3 4 5)
read -r -a SEED_GPUS <<< "${SEED_GPU_LIST_TEXT}"

if [[ ${#SEED_GPUS[@]} -ne ${#TRAIN_SEEDS[@]} ]]; then
    echo "JNCA_SEED_GPUS must contain exactly five GPU ids, one for each training seed." >&2
    exit 1
fi

case "${VARIANT}" in
    no_trust_fixed05)
        METHOD_DIR="egat_no_truststate"
        EXPERIMENT_DIR="fixed05"
        INCLUDE_TRUST="false"
        SECURITY_WEIGHT="0.5"
        ;;
    trust_fixed05)
        METHOD_DIR="egat_truststate"
        EXPERIMENT_DIR="fixed05"
        INCLUDE_TRUST="true"
        SECURITY_WEIGHT="0.5"
        ;;
    trust_ssi_adaptive)
        METHOD_DIR="egat_truststate"
        EXPERIMENT_DIR="ssi_adaptive"
        INCLUDE_TRUST="true"
        SECURITY_WEIGHT="-1.0"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}" >&2
        echo "Choose: no_trust_fixed05, trust_fixed05, trust_ssi_adaptive" >&2
        exit 1
        ;;
esac

TRAIN_ROOT="${BASE_RUN_ROOT}/retraining/${EXPERIMENT_DIR}/training"
LOG_ROOT="${BASE_RUN_ROOT}/retraining/${EXPERIMENT_DIR}/logs"

for required_path in "${VSETTING}" "${P_NET}" "${VALIDATION_ROOT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if ! grep -Eq '^[[:space:]]*lam:[[:space:]]*0\.15([[:space:]]|$)' "${VSETTING}"; then
    echo "The training setting does not contain arrival_rate lam: 0.15: ${VSETTING}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${TRAIN_ROOT}/${METHOD_DIR}" "${LOG_ROOT}"

run_seed() {
    local index="$1"
    local train_seed="${TRAIN_SEEDS[$index]}"
    local gpu="${SEED_GPUS[$index]}"
    local output_dir="${TRAIN_ROOT}/${METHOD_DIR}/train_seed_${train_seed}"
    local log_file="${LOG_ROOT}/${METHOD_DIR}_train_seed_${train_seed}_gpu_${gpu}.log"

    if find "${output_dir}" -type f -name best_model.pkl -print -quit 2>/dev/null | grep -q .; then
        echo "A best checkpoint already exists; refusing to mix another run into: ${output_dir}" >&2
        return 1
    fi

    mkdir -p "${output_dir}"
    echo "[$(date '+%F %T')] START variant=${VARIANT} train=${train_seed} gpu=${gpu}"

    CUDA_VISIBLE_DEVICES="${gpu}" python -u main.py \
        --solver_name gat_seq2seq_ppo \
        --seed "${train_seed}" \
        --use_cuda true \
        --cuda_id 0 \
        --p_net_dataset_path "${P_NET}" \
        --v_sim_setting_path "${VSETTING}" \
        --validation_v_nets_dataset_path "${VALIDATION_ROOT}" \
        --regenerate_v_nets true \
        --shortest_method bfs_shortest \
        --k_shortest 1 \
        --normalize_ts true \
        --incremental_security_shaping true \
        --security_reward_weight "${SECURITY_WEIGHT}" \
        --reward_failure_penalty 1.0 \
        --include_trust_in_state "${INCLUDE_TRUST}" \
        --num_train_epochs 140 \
        --validation_interval 10 \
        --num_epochs 1 \
        --train_only true \
        --run_dir "${output_dir}" \
        > "${log_file}" 2>&1

    local config_path
    local checkpoint_path
    config_path="$(find "${output_dir}" -type f -name config.yaml -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    checkpoint_path="$(find "${output_dir}" -type f -name best_model.pkl -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

    if [[ -z "${config_path}" || -z "${checkpoint_path}" ]]; then
        echo "Training completed without config.yaml or best_model.pkl; inspect ${log_file}" >&2
        return 1
    fi
    grep -Eq "^include_trust_in_state: ${INCLUDE_TRUST}$" "${config_path}" || {
        echo "Config verification failed for include_trust_in_state: ${config_path}" >&2
        return 1
    }
    grep -Eq "^security_reward_weight: ${SECURITY_WEIGHT}$" "${config_path}" || {
        echo "Config verification failed for security_reward_weight: ${config_path}" >&2
        return 1
    }

    echo "[$(date '+%F %T')] DONE variant=${VARIANT} train=${train_seed} gpu=${gpu}"
    echo "Verified config: ${config_path}"
    echo "Checkpoint: ${checkpoint_path}"
}

pids=()
for index in "${!TRAIN_SEEDS[@]}"; do
    # Each seed is an independent process. Duplicate GPU ids intentionally run
    # concurrently on the same card when the user supplies such a mapping.
    run_seed "${index}" &
    pids+=("$!")
    echo "${VARIANT}: train seed ${TRAIN_SEEDS[$index]}, GPU ${SEED_GPUS[$index]}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

if [[ ${status} -ne 0 ]]; then
    echo "${VARIANT} training failed; inspect ${LOG_ROOT}." >&2
    exit ${status}
fi

echo "${VARIANT} training completed under ${TRAIN_ROOT}/${METHOD_DIR}"
