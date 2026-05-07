#!/bin/bash

for x in 0_2 0_8 1 2 5 11 22 45; do
    uv run python ./datasets_generator/bl2_to_bin_converter.py \
        --input-dir "D:/eval_cvpr2026/data/denoised_transformer_bl2/sync_${x}" \
        --output-dir "D:/eval_cvpr2026/data/denoised_transformer_bin/sync_${x}"
done
