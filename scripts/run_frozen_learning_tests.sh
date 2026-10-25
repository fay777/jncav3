#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

data_root="${JNCA_BENCHMARK_ROOT:-benchmark_jnca_v1}"
run_root="${JNCA_TEST_RUN_ROOT:-runs_formal_20261025_learning_test}"
gpu_list_text="${JNCA_TEST_GPUS:-2 3 4 5 6}"
read -r -a gpu_ids <<< "${gpu_list_text}"
train_seeds=(1 2 3 4 5)
test_seeds=(20001 20002 20003 20004 20005)

if [[ ${#gpu_ids[@]} -ne ${#train_seeds[@]} ]]; then
    echo "JNCA_TEST_GPUS must contain exactly five GPU ids." >&2
    exit 1
fi
if [[ ! -d "${data_root}/p_net" || ! -d "${data_root}/test/seed_20001" ]]; then
    echo "Frozen benchmark is missing under ${data_root}." >&2
    exit 1
fi

mkdir -p "${run_root}/logs"

run_family() {
    local tag="$1"
    local checkpoint_root="$2"
    local solver="$3"
    shift 3
    local -a method_flags=("$@")
    local -a family_pids=()

    for index in "${!train_seeds[@]}"; do
        local train_seed="${train_seeds[$index]}"
        local gpu_id="${gpu_ids[$index]}"
        local checkpoint
        checkpoint="$(find "${checkpoint_root}" -path "*/train_seed${train_seed}_*/*" -name best_model.pkl -type f -print -quit)"
        if [[ -z "${checkpoint}" ]]; then
            echo "Missing checkpoint for ${tag}, training seed ${train_seed}." >&2
            exit 1
        fi

        (
            for test_seed in "${test_seeds[@]}"; do
                output_dir="${run_root}/${tag}/train_seed${train_seed}/test_seed${test_seed}"
                CUDA_VISIBLE_DEVICES="${gpu_id}" python -u main.py \
                    --solver_name "${solver}" \
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
                    --if_save_records true \
                    --if_temp_save_records true \
                    --lcb_z_score 1.645 \
                    --run_dir "${output_dir}" \
                    "${method_flags[@]}"
            done
        ) > "${run_root}/logs/${tag}_trainseed${train_seed}.log" 2>&1 &
        family_pids+=("$!")
        echo "${tag}: train seed ${train_seed}, GPU ${gpu_id}, PID $!"
    done

    local status=0
    for pid in "${family_pids[@]}"; do
        wait "${pid}" || status=1
    done
    if [[ ${status} -ne 0 ]]; then
        echo "${tag} failed; inspect ${run_root}/logs." >&2
        exit ${status}
    fi
}

run_family \
    egat_no_truststate \
    runs_formal_20261023_fixedmix_corrected \
    gat_seq2seq_ppo \
    --normalize_ts true \
    --incremental_security_shaping true \
    --security_reward_weight 0.5 \
    --reward_failure_penalty 1.0 \
    --include_trust_in_state false

run_family \
    egat_truststate \
    runs_formal_20261023_fixedmix_truststate_corrected \
    gat_seq2seq_ppo \
    --normalize_ts true \
    --incremental_security_shaping true \
    --security_reward_weight 0.5 \
    --reward_failure_penalty 1.0 \
    --include_trust_in_state true

run_family \
    a2c_gcn \
    runs_formal_20261022/a2c_gcn \
    a2c_gcn_seq2seq \
    --normalize_ts false \
    --incremental_security_shaping false \
    --include_trust_in_state false

run_family \
    twostage_gcn_drl \
    runs_formal_20261022/two_stage_gcn_drl_adapted \
    two_stage_gcn_a2c_adapted \
    --normalize_ts false \
    --incremental_security_shaping false \
    --include_trust_in_state false

echo "All frozen learning-method tests completed under ${run_root}."
