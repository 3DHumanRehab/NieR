#!/bin/bash

# ours for all scenes
# export CUDA_VISIBLE_DEVICES=2


task="ablation_ld"
datasets=("dataset1" "dataset2" "dataset3")
work_dir=/
data_dir=$work_dir/dataset

count=0

model_dir=$work_dir/model/ablation/wo_ld
for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 0  --use_cosine 0 --use_specular 0 --ip "127.1.3.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 0  --use_cosine 0  --use_specular 0 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

model_dir=$work_dir/model/ablation/ld_wo_cosine_specular
for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 1  --use_cosine 0 --use_specular 0 --ip "127.1.3.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 1  --use_cosine 0  --use_specular 0 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

model_dir=$work_dir/model/ablation/ld_wo_cosine
for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 1  --use_cosine 0 --use_specular 1 --ip "127.1.3.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 1  --use_cosine 0  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

model_dir=$work_dir/model/ablation/ld_wo_specular
for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 0 --ip "127.1.3.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 0 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done

model_dir=$work_dir/model/ablation/ld_full
for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 --ip "127.1.3.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/$task/$dataset > $model_dir/$task/$dataset/metrics.log 2>&1 &
    wait
done
