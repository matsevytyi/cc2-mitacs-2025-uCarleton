#!/usr/bin/env bash
set -e

# ---------------- STAGE 1: Action Space Core (Run this first!) ----------------
# Sweeps top_k in {2, 4, 8, 16} x focus_mode in {stateful, salience}
for K in 2 4 8 16; do
    for FOCUS in stateful salience; do
        ./runner.sh PPO "$K" "$FOCUS" 2.0 None 2 64
    done
done

# ---------------- STAGE 2: Distribution & CLS Scaling -------------------------
# TODO (Wassi): Check TensorBoard from Stage 1, set BEST_K and BEST_FOCUS below,
# then uncomment and run Stage 2a, followed by Stage 2b.

# BEST_K=8
# BEST_FOCUS=stateful

# # Stage 2a: Sweep dist_lambda
# for LAMBDA in 0.5 1.0 2.0 4.0; do
#     ./runner.sh PPO "$BEST_K" "$BEST_FOCUS" "$LAMBDA" None 2 64
# done

# # Stage 2b: Set BEST_LAMBDA from 2a, then sweep cls_alpha
# BEST_LAMBDA=2.0
# for ALPHA in 0.5 1.0 None 4.0; do
#     ./runner.sh PPO "$BEST_K" "$BEST_FOCUS" "$BEST_LAMBDA" "$ALPHA" 2 64
# done

# ---------------- STAGE 3: Encoder Capacity -----------------------------------
# TODO (Wassi): Set BEST_ALPHA from Stage 2b, then uncomment and run:

# BEST_ALPHA=None
# for LAYERS in 1 2 4; do
#     for DIM in 32 64 128; do
#         ./runner.sh PPO "$BEST_K" "$BEST_FOCUS" "$BEST_LAMBDA" "$BEST_ALPHA" "$LAYERS" "$DIM"
#     done
# done
