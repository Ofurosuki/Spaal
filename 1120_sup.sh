#!/bin/bash
# uv run python HFR_Denoise/pipeline/baseline_pipeline.py \
#     --input-path "/data2/yoshida/1121_data" \
#     --output-path "/data2/yoshida/1121_denoised" \
#     --ckpt-path "HFR_Denoise/run/1120_dn_dim32_attn_deg_0.pt" \
#     --num-frames 81 \
#     --chunk-size 3 \
#     --spatial-chunk-size 4096 \
#     --mask-expansion 5 \
#     --use-d-attn \

uv run python HFR_Denoise/src/baseline_eval.py \
    --ckpt "HFR_Denoise/run/1120_dn_dim32_attn_deg_0.pt" \
    --data_root "/data2/yoshida/1121_data" \

