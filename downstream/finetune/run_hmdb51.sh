list_file=dataset/hmdb51_frame128_train_list.txt
eva_list_file=dataset/hmdb51_frame128_test_list.txt
root_path=/path/to/data
num_classes=51
num_gpu=4
pretrained_model=/path/to/model

python3 -m torch.distributed.launch --nproc_per_node=$num_gpu train.py \
--list-file=$list_file \
--root-path=$root_path \
--num-classes=$num_classes \
--pretrained-model=$pretrained_model


python3 -m torch.distributed.launch --nproc_per_node=$num_gpu eval.py \
--list-file=$eva_list_file \
--root-path=$root_path \
--num-classes=$num_classes \
--num-gpu=$num_gpu \
--pretrained-model=output/current.pth


