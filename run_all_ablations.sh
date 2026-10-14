#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

gpu_id="${1:-0}"
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    printf 'GPU ID must be a non-negative integer.\n' >&2
    exit 2
fi

experiments=(pcnet cph rspr cph_rspr)
seeds=(42 2026 4536)

# Refuse to begin if a later run would collide with an existing output.
for experiment in "${experiments[@]}"; do
    for seed in "${seeds[@]}"; do
        run_dir="outputs/${experiment}/seed_${seed}"
        if [[ -e "$run_dir" ]]; then
            printf 'Output already exists: %s\n' "$run_dir" >&2
            printf 'Inspect existing runs before starting; this script will not overwrite them.\n' >&2
            exit 1
        fi
    done
done

python verify_environment.py --training-only

# Each branch completes one real update before the 12 long runs begin.
smoke_name="smoke_preflight_$(date +%Y%m%d_%H%M%S)_$$"
for experiment in "${experiments[@]}"; do
    printf '\n[preflight] %s on GPU %s\n' "$experiment" "$gpu_id"
    python "train_${experiment}.py" \
        --gpu-id "$gpu_id" \
        --seed 42 \
        --smoke-test \
        --run-name "$smoke_name"
done

run_number=0
for experiment in "${experiments[@]}"; do
    for seed in "${seeds[@]}"; do
        run_number=$((run_number + 1))
        printf '\n[run %d/12] %s seed=%s GPU=%s\n' \
            "$run_number" "$experiment" "$seed" "$gpu_id"
        python "train_${experiment}.py" --gpu-id "$gpu_id" --seed "$seed"
    done
done

printf '\nAll 12 ablation runs completed.\n'
