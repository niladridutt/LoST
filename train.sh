# Semanticist training
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_old.yaml
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l.yaml

# aws s3 cp --recursive "s3://phidias/niladri/direct3d_latents" /mnt/localssd/direct3d_latents/train
# aws s3 cp --recursive "s3://phidias/niladri/flux_images" /mnt/localssd/flux_images
# python split_train_val.py  

# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/eg.yaml
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/autoregressive_eg.yaml
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_mini.yaml
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_spatial_from_no.yaml

# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_w_32.yaml
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_1x1.yaml

# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l_2x2_resume.yaml

# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_xl.yaml
# # or use torchrun
# torchrun --nproc-per-node=8 train_net.py --cfg configs/tokenizer_xl.yaml
# # or use submitit
# python submitit_train.py --ngpus=8 --nodes=1 --partition=xxx --config configs/tokenizer_xl.yaml

# # ϵLlamaGen training
# accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/autoregressive_xl.yaml
# # or use torchrun
# torchrun --nproc-per-node=8 train_net.py --cfg configs/autoregressive_xl.yaml
# # or use submitit
# python submitit_train.py --ngpus=8 --nodes=1 --partition=xxx --config configs/autoregressive_xl.yaml
