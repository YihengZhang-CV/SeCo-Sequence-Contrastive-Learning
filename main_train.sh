#!/bin/bash

data_dir="/dataset/Kinetics_t30_s256"
output_dir="./model"
eval_dir="./model/eval_svm"
pretrained="pretrain/moco_v2_200ep_pretrain.pth.tar"
num_replica=4

mkdir -p ${output_dir}
mkdir -p ${eval_dir}

python3 -m torch.distributed.launch --master_port 12857 --nproc_per_node=${num_replica} \
    train_inter_intra_order.py \
    --data_dir=${data_dir} \
    --datasplit=train_list \
    --pretrained_model=${pretrained} \
    --output_dir=${output_dir} \
    --model_mlp