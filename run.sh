#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash run.sh <gpu> <method> <metric>" >&2
    echo "Example: bash run.sh 0 tpt cas" >&2
    exit 2
fi

gpu=$1
method=$2
metric=$3

case "$metric" in
    cas|energy|mcm|random) ;;
    *) echo "Unsupported metric: $metric" >&2; exit 2 ;;
esac

batch_size=64
context_init=a_photo_of_a
data_root=${DATA_ROOT:-./datasets}
output_dir=${OUTPUT_DIR:-results}

datasets=(
    DTD Aircraft Flower102 Caltech101 eurosat Pets UCF101 Cars
    SUN397 Food101 A V R I K
)

for dataset in "${datasets[@]}"; do
    python instance_metric.py \
        --test_sets "$dataset" \
        --batch-size "$batch_size" \
        --ctx_init "$context_init" \
        --print-freq 50 \
        --output_dir "$output_dir" \
        --algorithm "$method" \
        --gpu "$gpu" \
        --arch ViT-B/16 \
        --data "$data_root" \
        --metric "$metric"
done
