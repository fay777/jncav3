#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

data_root="${JNCA_BENCHMARK_ROOT:-benchmark_jnca_v1}"
run_root="${JNCA_ABLATION_TRAIN_ROOT:-runs_ablation_20261028_training}"
gpu_list_text="${JNCA_ABLATION_GPUS:-2 3 4 5 6}"
variant_list_text="${JNCA_ABLATION_VARIANTS:-no_incremental_shaping no_security_reward no_efficiency_reward}"
read -r -a gpu_ids <<< "${gpu_list_text}"
read -r -a variants <<< "${variant_list_text}"
train_seeds=(1 2 3 4 5)

if [[ ${#gpu_ids[@]} -ne ${#train_seeds[@]} ]]; then
    echo "JNCA_ABLATION_GPUS must contain exactly five GPU ids." >&2
    exit 1
fi
if [[ ! -f "${data_root}/p_net/p_net.gml" ]]; then
    echo "Frozen physical network is missing: ${data_root}/p_net/p_net.gml" >&2
    exit 1
fi
if [[ ! -f "${data_root}/validation/seed_10001/events.yaml" ]]; then
    echo "Frozen validation set is missing: ${data_root}/validation/seed_10001" >&2
    exit 1
fi
if [[ -e "${run_root}" ]]; then
    echo "Ablation training directory already exists; refusing to overwrite: ${run_root}" >&2
    exit 1
fi

mkdir -p "${run_root}/logs"
printf 'variant,normalize_ts,incremental_security_shaping,terminal_only_security_reward,security_reward_weight,include_trust_in_state,graph_encoder,use_lstm,reward_failure_penalty,num_train_epochs,validation_interval,experiment_role\n' \
    > "${run_root}/ablation_config.csv"

variant_flags() {
    local variant="$1"
    case "${variant}" in
        no_incremental_shaping)
            flags=(
                --normalize_ts true
                --incremental_security_shaping false
                --terminal_only_security_reward true
                --security_reward_weight 0.5
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm true
            )
            config_row="no_incremental_shaping,true,false,true,0.5,true,egat,true,1.0,140,10,core_component_ablation"
            ;;
        adaptive_security_weight)
            flags=(
                --normalize_ts true
                --incremental_security_shaping true
                --terminal_only_security_reward false
                --security_reward_weight -1.0
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm true
            )
            config_row="adaptive_security_weight,true,true,false,-1.0,true,egat,true,1.0,140,10,supplementary_weight_strategy"
            ;;
        no_security_reward)
            flags=(
                --normalize_ts true
                --incremental_security_shaping true
                --terminal_only_security_reward false
                --security_reward_weight 0.0
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm true
            )
            config_row="no_security_reward,true,true,false,0.0,true,egat,true,1.0,140,10,core_component_ablation"
            ;;
        no_efficiency_reward)
            flags=(
                --normalize_ts true
                --incremental_security_shaping true
                --terminal_only_security_reward false
                --security_reward_weight 1.0
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm true
            )
            config_row="no_efficiency_reward,true,true,false,1.0,true,egat,true,1.0,140,10,core_component_ablation"
            ;;
        egat_to_gcn)
            flags=(
                --normalize_ts true
                --incremental_security_shaping true
                --terminal_only_security_reward false
                --security_reward_weight 0.5
                --include_trust_in_state true
                --graph_encoder gcn
                --use_lstm true
            )
            config_row="egat_to_gcn,true,true,false,0.5,true,gcn,true,1.0,140,10,core_architecture_ablation"
            ;;
        without_lstm)
            flags=(
                --normalize_ts true
                --incremental_security_shaping true
                --terminal_only_security_reward false
                --security_reward_weight 0.5
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm false
            )
            config_row="without_lstm,true,true,false,0.5,true,egat,false,1.0,140,10,core_architecture_ablation"
            ;;
        legacy_reward)
            flags=(
                --normalize_ts false
                --incremental_security_shaping false
                --terminal_only_security_reward false
                --security_reward_weight 0.5
                --include_trust_in_state true
                --graph_encoder egat
                --use_lstm true
            )
            config_row="legacy_reward,false,false,false,0.5,true,egat,true,1.0,140,10,supplementary_combined_rollback"
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
    printf '%s\n' "${config_row}" >> "${run_root}/ablation_config.csv"

    for index in "${!train_seeds[@]}"; do
        local train_seed="${train_seeds[$index]}"
        local gpu_id="${gpu_ids[$index]}"
        local output_dir="${run_root}/${variant}/train_seed${train_seed}_gpu${gpu_id}"
        local log_file="${run_root}/logs/${variant}_trainseed${train_seed}_gpu${gpu_id}.log"

        CUDA_VISIBLE_DEVICES="${gpu_id}" python -u main.py \
            --solver_name gat_seq2seq_ppo \
            --seed "${train_seed}" \
            --use_cuda true \
            --cuda_id 0 \
            --p_net_dataset_path "${data_root}/p_net" \
            --validation_v_nets_dataset_path "${data_root}/validation/seed_10001" \
            --shortest_method bfs_shortest \
            --k_shortest 1 \
            --reusable false \
            --allow_rejection false \
            --reward_failure_penalty 1.0 \
            --num_train_epochs 140 \
            --validation_interval 10 \
            --train_only true \
            --if_save_records true \
            --if_temp_save_records true \
            --run_dir "${output_dir}" \
            "${flags[@]}" \
            > "${log_file}" 2>&1 &

        variant_pids+=("$!")
        echo "${variant}: train seed ${train_seed}, GPU ${gpu_id}, PID $!"
    done

    local status=0
    for pid in "${variant_pids[@]}"; do
        wait "${pid}" || status=1
    done
    if [[ ${status} -ne 0 ]]; then
        echo "${variant} failed; inspect ${run_root}/logs." >&2
        exit ${status}
    fi

    local checkpoint_count
    checkpoint_count="$(find "${run_root}/${variant}" -type f -name best_model.pkl | wc -l)"
    if [[ "${checkpoint_count}" -ne 5 ]]; then
        echo "${variant} completed but produced ${checkpoint_count} best checkpoints instead of 5." >&2
        exit 1
    fi
    echo "${variant}: all five training seeds completed."
}

for variant in "${variants[@]}"; do
    run_variant "${variant}"
done

echo "All requested ablation training variants completed under ${run_root}."
