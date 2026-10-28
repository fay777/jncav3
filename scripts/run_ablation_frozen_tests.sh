#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

data_root="${JNCA_BENCHMARK_ROOT:-benchmark_jnca_v1}"
train_root="${JNCA_ABLATION_TRAIN_ROOT:-runs_ablation_20261028_training}"
test_root="${JNCA_ABLATION_TEST_ROOT:-runs_ablation_20261028_frozen_test}"
gpu_list_text="${JNCA_ABLATION_GPUS:-2 3 4 5 6}"
variant_list_text="${JNCA_ABLATION_VARIANTS:-no_incremental_shaping adaptive_security_weight no_security_reward legacy_reward}"
read -r -a gpu_ids <<< "${gpu_list_text}"
read -r -a variants <<< "${variant_list_text}"
train_seeds=(1 2 3 4 5)
test_seeds=(20001 20002 20003 20004 20005)

if [[ ${#gpu_ids[@]} -ne ${#train_seeds[@]} ]]; then
    echo "JNCA_ABLATION_GPUS must contain exactly five GPU ids." >&2
    exit 1
fi
if [[ ! -f "${data_root}/p_net/p_net.gml" ]]; then
    echo "Frozen physical network is missing: ${data_root}/p_net/p_net.gml" >&2
    exit 1
fi
for test_seed in "${test_seeds[@]}"; do
    if [[ ! -f "${data_root}/test/seed_${test_seed}/events.yaml" ]]; then
        echo "Frozen test set is missing: ${data_root}/test/seed_${test_seed}" >&2
        exit 1
    fi
done
if [[ -e "${test_root}" ]]; then
    echo "Ablation test directory already exists; refusing to overwrite: ${test_root}" >&2
    exit 1
fi

mkdir -p "${test_root}/logs"
printf 'variant,train_seed,checkpoint_path,checkpoint_sha256,checkpoint_size_bytes\n' \
    > "${test_root}/checkpoint_manifest.csv"

variant_flags() {
    local variant="$1"
    case "${variant}" in
        no_incremental_shaping)
            flags=(--normalize_ts true --incremental_security_shaping false --security_reward_weight 0.5 --include_trust_in_state true)
            ;;
        adaptive_security_weight)
            flags=(--normalize_ts true --incremental_security_shaping true --security_reward_weight -1.0 --include_trust_in_state true)
            ;;
        no_security_reward)
            flags=(--normalize_ts true --incremental_security_shaping true --security_reward_weight 0.0 --include_trust_in_state true)
            ;;
        legacy_reward)
            flags=(--normalize_ts false --incremental_security_shaping false --security_reward_weight 0.5 --include_trust_in_state true)
            ;;
        *)
            echo "Unknown ablation variant: ${variant}" >&2
            exit 1
            ;;
    esac
}

run_variant() {
    local variant="$1"
    local -a variant_pids=()
    variant_flags "${variant}"

    for index in "${!train_seeds[@]}"; do
        local train_seed="${train_seeds[$index]}"
        local gpu_id="${gpu_ids[$index]}"
        local -a matches=()
        mapfile -t matches < <(
            find "${train_root}/${variant}" \
                -path "*/train_seed${train_seed}_*/*" \
                -type f \
                -name best_model.pkl \
                -print
        )
        if [[ ${#matches[@]} -ne 1 ]]; then
            echo "Expected one ${variant} checkpoint for train seed ${train_seed}; found ${#matches[@]}." >&2
            printf '%s\n' "${matches[@]}" >&2
            exit 1
        fi

        local checkpoint
        local checkpoint_sha
        local checkpoint_size
        checkpoint="$(readlink -f "${matches[0]}")"
        checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
        checkpoint_size="$(stat -c '%s' "${checkpoint}")"
        printf '%s,%s,%s,%s,%s\n' \
            "${variant}" "${train_seed}" "${checkpoint}" "${checkpoint_sha}" "${checkpoint_size}" \
            >> "${test_root}/checkpoint_manifest.csv"

        (
            for test_seed in "${test_seeds[@]}"; do
                output_dir="${test_root}/${variant}/train_seed${train_seed}/test_seed${test_seed}"
                CUDA_VISIBLE_DEVICES="${gpu_id}" python -u main.py \
                    --solver_name gat_seq2seq_ppo \
                    --seed "${train_seed}" \
                    --use_cuda true \
                    --cuda_id 0 \
                    --p_net_dataset_path "${data_root}/p_net" \
                    --v_nets_dataset_path "${data_root}/test/seed_${test_seed}" \
                    --pretrained_model_path "${checkpoint}" \
                    --num_train_epochs 0 \
                    --num_epochs 1 \
                    --train_only false \
                    --regenerate_v_nets false \
                    --shortest_method bfs_shortest \
                    --k_shortest 1 \
                    --reusable false \
                    --allow_rejection false \
                    --reward_failure_penalty 1.0 \
                    --if_save_records true \
                    --if_temp_save_records true \
                    --lcb_z_score 1.645 \
                    --run_dir "${output_dir}" \
                    "${flags[@]}"
            done
        ) > "${test_root}/logs/${variant}_trainseed${train_seed}.log" 2>&1 &

        variant_pids+=("$!")
        echo "${variant}: test train-seed ${train_seed}, GPU ${gpu_id}, PID $!"
    done

    local status=0
    for pid in "${variant_pids[@]}"; do
        wait "${pid}" || status=1
    done
    if [[ ${status} -ne 0 ]]; then
        echo "${variant} frozen testing failed; inspect ${test_root}/logs." >&2
        exit ${status}
    fi

    local summary_count
    summary_count="$(find "${test_root}/${variant}" -type f -name global_summary.csv | wc -l)"
    if [[ "${summary_count}" -ne 25 ]]; then
        echo "${variant} produced ${summary_count} summaries instead of 25." >&2
        exit 1
    fi
    echo "${variant}: all 25 frozen tests completed."
}

for variant in "${variants[@]}"; do
    run_variant "${variant}"
done

echo "All requested ablation frozen tests completed under ${test_root}."
