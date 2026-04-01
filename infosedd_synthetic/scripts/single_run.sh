export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=5
cd infosedd-synthetic
python -m train --config-name config.yaml data=high_dim estimator=infosedd_j ++mutual_information=1