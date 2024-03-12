#!/bin/bash

# ours for all scenes
# export CUDA_VISIBLE_DEVICES=2   


task="ours"
datasets=("dataset1" "dataset2" "dataset3")
work_dir=/
data_dir=$work_dir/dataset
model_dir=$work_dir/model

count=0

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 1.5 2 2.5 3 3.5  \
   --use_hierarchical_split 1 --densify_split_N 3  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.2.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

