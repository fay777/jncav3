#!/usr/bin/env bash
# Zero-shot robustness tests: existing EGAT and GCN checkpoints only.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

: "${JNCA_BENCHMARK_ROOT:?Set to the original benchmark_jnca_v1 directory.}"
: "${JNCA_LINK_HETERO_ROOT:?Set to the output root created by create_link_heterogeneity_benchmarks.py.}"
: "${JNCA_HETERO_FULL_TRAIN_ROOT:?Set to the existing Full EGAT-DPOT checkpoint root.}"
: "${JNCA_HETERO_GCN_TRAIN_ROOT:?Set to the existing EGAT-to-GCN checkpoint root.}"

test_root="${JNCA_HETERO_TEST_ROOT:-runs_link_heterogeneity_20261105}"
gpu_list_text="${JNCA_HETERO_GPUS:-2 3 4 5 6}"
variant_list_text="${JNCA_HETERO_VARIANTS:-bw_hetero security_hetero joint_hetero}"
read -r -a gpu_ids <<< "${gpu_list_text}"
read -r -a link_variants <<< "${variant_list_text}"
train_seeds=(1 2 3 4 5)
test_seeds=(20001 20002 20003 20004 20005)

if [[ ${#gpu_ids[@]} -ne 5 ]]; then
    echo 'JNCA_HETERO_GPUS must contain exactly five GPU ids.' >&2
    exit 1
fi
if [[ -e "${test_root}" ]]; then
    echo "Result directory already exists; refusing to overwrite: ${test_root}" >&2
    exit 1
fi
for test_seed in "${test_seeds[@]}"; do
    [[ -f "${JNCA_BENCHMARK_ROOT}/test/seed_${test_seed}/events.yaml" ]] || {
        echo "Missing frozen test set: seed_${test_seed}" >&2; exit 1; }
done
for link_variant in "${link_variants[@]}"; do
    [[ -f "${JNCA_LINK_HETERO_ROOT}/${link_variant}/p_net/p_net.gml" ]] || {
        echo "Missing heterogeneous p_net: ${link_variant}" >&2; exit 1; }
done

mkdir -p "${test_root}/logs"
printf 'link_variant,method,train_seed,checkpoint_path,checkpoint_sha256\n' > "${test_root}/checkpoint_manifest.csv"

checkpoint_for() {
    local root="$1" seed="$2"
    local -a matches=()
    mapfile -t matches < <(find "${root}" -path "*/train_seed${seed}_*/*" -type f -name best_model.pkl -print)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected one checkpoint for train seed ${seed} in ${root}; found ${#matches[@]}." >&2
        return 1
    fi
    readlink -f "${matches[0]}"
}

run_method() {
    local link_variant="$1" method="$2" train_root="$3" graph_encoder="$4"
    local -a pids=()
    for index in "${!train_seeds[@]}"; do
        local train_seed="${train_seeds[$index]}" gpu_id="${gpu_ids[$index]}"
        local checkpoint
        checkpoint="$(checkpoint_for "${train_root}" "${train_seed}")"
        printf '%s,%s,%s,%s,%s\n' "${link_variant}" "${method}" "${train_seed}" "${checkpoint}" \
            "$(sha256sum "${checkpoint}" | awk '{print $1}')" >> "${test_root}/checkpoint_manifest.csv"
        (
            for test_seed in "${test_seeds[@]}"; do
                CUDA_VISIBLE_DEVICES="${gpu_id}" python -u main.py \
                    --solver_name gat_seq2seq_ppo --seed "${train_seed}" \
                    --use_cuda true --cuda_id 0 \
                    --p_net_dataset_path "${JNCA_LINK_HETERO_ROOT}/${link_variant}/p_net" \
                    --v_nets_dataset_path "${JNCA_BENCHMARK_ROOT}/test/seed_${test_seed}" \
                    --pretrained_model_path "${checkpoint}" \
                    --num_train_epochs 0 --num_epochs 1 --train_only false --regenerate_v_nets false \
                    --shortest_method bfs_shortest --k_shortest 1 --reusable false --allow_rejection false \
                    --normalize_ts true --incremental_security_shaping true --terminal_only_security_reward false \
                    --security_reward_weight 0.5 --include_trust_in_state true \
                    --graph_encoder "${graph_encoder}" --use_lstm true --reward_failure_penalty 1.0 \
                    --if_save_records true --if_temp_save_records true --lcb_z_score 1.645 \
                    --run_dir "${test_root}/${link_variant}/${method}/train_seed${train_seed}/test_seed${test_seed}"
            done
        ) > "${test_root}/logs/${link_variant}_${method}_trainseed${train_seed}.log" 2>&1 &
        pids+=("$!")
        echo "${link_variant}/${method}: seed ${train_seed}, GPU ${gpu_id}, PID $!"
    done
    local status=0
    for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
    [[ ${status} -eq 0 ]] || return "${status}"
}

for link_variant in "${link_variants[@]}"; do
    run_method "${link_variant}" full_egat "${JNCA_HETERO_FULL_TRAIN_ROOT}" egat
    run_method "${link_variant}" topology_gcn "${JNCA_HETERO_GCN_TRAIN_ROOT}" gcn
done

expected=$(( ${#link_variants[@]} * 2 * 5 * 5 ))
actual="$(find "${test_root}" -type f -name global_summary.csv | wc -l)"
[[ "${actual}" -eq "${expected}" ]] || { echo "Expected ${expected} summaries, got ${actual}." >&2; exit 1; }
echo "Completed ${actual} frozen zero-shot tests in ${test_root}."
