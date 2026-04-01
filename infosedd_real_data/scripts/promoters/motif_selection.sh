#!/bin/bash

current_date=$(date +"%Y_%m_%d")

mkdir -p "output_logs/${current_date}"

python main.py --config-name=config_motif_finder \
    mode=motif_selection \
    data_path=??? \
    variant=j \
    lower_bound=140 \
    upper_bound=180\
    variant_c_target=0 \
    ++mask_type=keep \
    noise=loglinear \
    model=caduceus1k \
    ++box_length=6 \
    backbone=caduceus \
    parameterization=subs \
    strategy=single_device \
    data=genomic_plant_tata \
    eval.generate_samples=False \
    eval.checkpoint_path=??? \
    eval.mc_estimates=1 \
    loader.global_batch_size=1024 \
    loader.batch_size=1024 \
    loader.eval_batch_size=1024 \
    strategy.device=0 \
    data.p_random=0.0 \