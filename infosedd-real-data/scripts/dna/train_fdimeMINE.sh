#!/bin/bash
cd infosedd-real-data

for p_random in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      python main.py --config-name=config_genomics \
            mode=train \
            model=caduceus1k \
            +model.hidden_size=256 \
            lr_scheduler=constant \
            backbone=caduceus \
            parameterization=discriminative \
            eval.disable_ema=True \
            strategy=single_device \
            data=genomic \
            eval.generate_samples=False \
            eval.checkpoint_path=kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
            loader.global_batch_size=128 \
            optim.lr=1e-3 \
            alpha=1.0 \
            divergence=MINE \
            strategy.device=0 \
            data.p_random=$p_random \
            train_marginal=False
done