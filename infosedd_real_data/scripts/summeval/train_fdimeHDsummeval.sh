#!/bin/bash
cd infosedd-real-data

for p_random in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  python main.py \
    mode=train \
    backbone=hf_dit \
    strategy=single_device \
    data=summeval \
    parameterization=discriminative \
    divergence=HD \
    data.p_random=$p_random \
    alpha=1.0 \
    train_marginal=False \
    data_path=/home/foresti/mdlm/model_summaries_cleaned/M22/outputs_cnndm.aligned.paired.jsonl \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    loader.global_batch_size=16 \
    strategy.device=0 \
    eval.generate_samples=False \
    wandb.notes="Mine experiments p decaying" \
    time_conditioning=False \
    eval.disable_ema=True
done