#!/bin/bash

# baseline for all scenes
# export CUDA_VISIBLE_DEVICES=2   


task="baseline"
datasets=("dataset1" "dataset2" "dataset3")
work_dir=/
data_dir=$work_dir/dataset
model_dir=$work_dir/model

count=0

for dataset in "${datasets[@]}"
do
    ((count++))
    mkdir -p $model_dir/$task/$dataset
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 0 --ip "127.1.1.$count"  > $model_dir/$task/$dataset/train.log 2>&1 &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 0 > $model_dir/$task/$dataset/render.log 2>&1   &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

