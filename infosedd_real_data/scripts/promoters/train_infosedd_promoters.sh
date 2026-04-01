#!/bin/bash

python main.py --config-name=config_genomics \
    data_path=/home/foresti/infosedd-rebuttal-mdlm/Genomic_Arabidopsis_promoter_dataset_tata \
    mode=train \
    variant=j \
    noise=loglinear \
    model=caduceus1k \
    lr_scheduler=constant \
    ++trainer.check_val_every_n_epoch=1 \
    ++trainer.max_steps=100000 \
    backbone=caduceus \
    parameterization=subs \
    strategy=single_device \
    data=genomic_plant_tata \
    eval.generate_samples=False \
    eval.checkpoint_path=kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3 \
    loader.global_batch_size=128 \
    loader.batch_size=128 \
    loader.eval_batch_size=128 \
    optim.lr=1e-3 \
    strategy.device=1 \
    data.p_random=0.0 \