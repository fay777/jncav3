#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

manifest_path="${JNCA_CHECKPOINT_MANIFEST_PATH:-checkpoint_manifest.tsv}"
egat_no_trust_root="${JNCA_EGAT_NO_TRUST_ROOT:-runs_formal_20261023_fixedmix_corrected}"
egat_trust_root="${JNCA_EGAT_TRUST_ROOT:-runs_formal_20261023_fixedmix_truststate_corrected}"
a2c_root="${JNCA_A2C_ROOT:-runs_formal_20261022/a2c_gcn}"
two_stage_root="${JNCA_TWO_STAGE_ROOT:-runs_formal_20261022/two_stage_gcn_drl_adapted}"
train_seeds=(1 2 3 4 5)

mkdir -p "$(dirname "${manifest_path}")"
printf 'method\tsolver\ttrain_seed\tcheckpoint_path\tcheckpoint_sha256\tcheckpoint_size_bytes\n' > "${manifest_path}"

record_family() {
    local tag="$1"
    local checkpoint_root="$2"
    local solver="$3"

    if [[ ! -d "${checkpoint_root}" ]]; then
        echo "Checkpoint root does not exist for ${tag}: ${checkpoint_root}" >&2
        exit 1
    fi

    for train_seed in "${train_seeds[@]}"; do
        local -a matches=()
        mapfile -t matches < <(
            find "${checkpoint_root}" \
                -path "*/train_seed${train_seed}_*/*" \
                -name best_model.pkl \
                -type f \
                -print
        )
        if [[ ${#matches[@]} -ne 1 ]]; then
            echo "Expected exactly one checkpoint for ${tag}, training seed ${train_seed}; found ${#matches[@]}." >&2
            printf '%s\n' "${matches[@]}" >&2
            exit 1
        fi

        local checkpoint_abs
        local checkpoint_sha256
        local checkpoint_size
        checkpoint_abs="$(readlink -f "${matches[0]}")"
        checkpoint_sha256="$(sha256sum "${checkpoint_abs}" | awk '{print $1}')"
        checkpoint_size="$(stat -c '%s' "${checkpoint_abs}")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${tag}" \
            "${solver}" \
            "${train_seed}" \
            "${checkpoint_abs}" \
            "${checkpoint_sha256}" \
            "${checkpoint_size}" \
            >> "${manifest_path}"
    done
}

record_family egat_no_truststate "${egat_no_trust_root}" gat_seq2seq_ppo
record_family egat_truststate "${egat_trust_root}" gat_seq2seq_ppo
record_family a2c_gcn "${a2c_root}" a2c_gcn_seq2seq
record_family twostage_gcn_drl "${two_stage_root}" two_stage_gcn_a2c_adapted

echo "Checkpoint manifest saved to ${manifest_path}."
