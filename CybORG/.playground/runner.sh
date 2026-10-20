#!/usr/bin/env bash
set -e

# Usage: ./runner.sh <ALGO> <TOP_K> <FOCUS_MODE> <DIST_LAMBDA> <CLS_ALPHA> <N_LAYERS> <EMBED_DIM>
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALGO=${1:-PPO}
TOP_K=${2:-8}
FOCUS_MODE=${3:-stateful}
DIST_LAMBDA=${4:-2.0}
CLS_ALPHA=${5:-None}
N_LAYERS=${6:-2}
EMBED_DIM=${7:-64}

echo "========================================================"
echo "[1/2] STATIC TRAINING: ALGO=$ALGO | K=$TOP_K | FOCUS=$FOCUS_MODE | LAMBDA=$DIST_LAMBDA | ALPHA=$CLS_ALPHA | L=$N_LAYERS | D=$EMBED_DIM"
echo "========================================================"
python training_pipeline.py \
    --algo "$ALGO" \
    --mode transformer \
    --action_space_mode topk \
    --top_k "$TOP_K" \
    --focus_mode "$FOCUS_MODE" \
    --dist_lambda "$DIST_LAMBDA" \
    --cls_alpha "$CLS_ALPHA" \
    --n_layers "$N_LAYERS" \
    --embedding_dim "$EMBED_DIM"

python process_tensorboard_IEEE.py \
    --phase training \
    --algo "$ALGO" \
    --mode transformer \
    --top_k "$TOP_K" \
    --focus_mode "$FOCUS_MODE" \
    --dist_lambda "$DIST_LAMBDA" \
    --cls_alpha "$CLS_ALPHA" \
    --n_layers "$N_LAYERS" \
    --embedding_dim "$EMBED_DIM"

echo "========================================================"
echo "[2/2] DYNAMIC TUNING: ALGO=$ALGO | K=$TOP_K | FOCUS=$FOCUS_MODE | LAMBDA=$DIST_LAMBDA | ALPHA=$CLS_ALPHA | L=$N_LAYERS | D=$EMBED_DIM"
echo "========================================================"
python final_tuning_pipeline.py \
    --algo "$ALGO" \
    --mode transformer \
    --action_space_mode topk \
    --top_k "$TOP_K" \
    --focus_mode "$FOCUS_MODE" \
    --dist_lambda "$DIST_LAMBDA" \
    --cls_alpha "$CLS_ALPHA" \
    --n_layers "$N_LAYERS" \
    --embedding_dim "$EMBED_DIM"

python process_tensorboard_IEEE.py \
    --phase tuning \
    --algo "$ALGO" \
    --mode transformer \
    --top_k "$TOP_K" \
    --focus_mode "$FOCUS_MODE" \
    --dist_lambda "$DIST_LAMBDA" \
    --cls_alpha "$CLS_ALPHA" \
    --n_layers "$N_LAYERS" \
    --embedding_dim "$EMBED_DIM"
