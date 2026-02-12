export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3
cd infosedd-synthetic
python -m train --multirun --config-name config.yaml data=high_dim estimator=infosedd_c ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=fdime_hd ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=fdime_gan ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=fdime_smile ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=infosedd_j ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=minde data.config.normalize=True ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=fdime_mine ++mutual_information=10,20,30,40,50
python -m train --multirun --config-name config.yaml data=high_dim estimator=fdime_nwj ++mutual_information=10,20,30,40,50