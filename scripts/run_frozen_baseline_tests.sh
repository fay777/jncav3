#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

data_root="${JNCA_BENCHMARK_ROOT:-benchmark_jnca_v1}"
run_root="${JNCA_BASELINE_RUN_ROOT:-runs_formal_20261025_baselines_security_metrics}"
methods=(first_fit rw_rank_bfs mcts appf_dsi_adapted ssd_ose_adapted)
test_seeds=(20001 20002 20003 20004 20005)

if [[ ! -d "${data_root}/p_net" || ! -d "${data_root}/test/seed_20001" ]]; then
    echo "Frozen benchmark is missing under ${data_root}." >&2
    exit 1
fi
if [[ -e "${run_root}" ]]; then
    echo "Result directory already exists: ${run_root}" >&2
    exit 1
fi

mkdir -p "${run_root}/logs"
declare -a method_pids=()

for method in "${methods[@]}"; do
    (
        for test_seed in "${test_seeds[@]}"; do
            output_dir="${run_root}/${method}/test_seed${test_seed}"
            python -u main.py \
                --solver_name "${method}" \
                --seed "${test_seed}" \
                --use_cuda false \
                --p_net_dataset_path "${data_root}/p_net" \
                --v_nets_dataset_path "${data_root}/test/seed_${test_seed}" \
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
                --run_dir "${output_dir}"
        done
    ) > "${run_root}/logs/${method}.log" 2>&1 &
    method_pids+=("$!")
    echo "${method}: PID $!"
done

status=0
for pid in "${method_pids[@]}"; do
    wait "${pid}" || status=1
done
if [[ ${status} -ne 0 ]]; then
    echo "At least one baseline failed; inspect ${run_root}/logs." >&2
    exit ${status}
fi

echo "All frozen baseline tests completed under ${run_root}."
