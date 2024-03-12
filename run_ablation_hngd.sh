
#!/bin/bash

# ours for all scenes
# export CUDA_VISIBLE_DEVICES=2


task="ablation_hngd"
datasets=("dataset1" "dataset2" "dataset3")
work_dir=/
data_dir=$work_dir/dataset

count=0

model_dir=$work_dir/model/ablation/hngd_thre

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/1/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1  \
   --use_hierarchical_split 1 --densify_split_N 1  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/1/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/1/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/1/$task/$dataset > $model_dir/1/$task/$dataset/metrics.log 2>&1 &
    wait
done

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/123/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 2 3  \
   --use_hierarchical_split 1 --densify_split_N 2  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/123/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/123/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/123/$task/$dataset > $model_dir/123/$task/$dataset/metrics.log 2>&1 &
    wait
done

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/d123/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 1.5 2 2.5 3 3.5 \
   --use_hierarchical_split 1 --densify_split_N 3  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/d123/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/d123/$task/$dataset > $model_dir/d123/$task/$dataset/metrics.log 2>&1 &
    wait
done








model_dir=$work_dir/model/ablation/hngd_split_n

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/1/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1  \
   --use_hierarchical_split 1 --densify_split_N 1  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/1/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/1/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/1/$task/$dataset > $model_dir/1/$task/$dataset/metrics.log 2>&1 &
    wait
done

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/2/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 2 3  \
   --use_hierarchical_split 1 --densify_split_N 2  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/2/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/2/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/2/$task/$dataset > $model_dir/2/$task/$dataset/metrics.log 2>&1 &
    wait
done

for dataset in "${datasets[@]}"
do
    ((count++))
    nohup python train.py -s $data_dir/$dataset -m $model_dir/3/$task/$dataset   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 1.5 2 2.5 3 3.5 \
   --use_hierarchical_split 1 --densify_split_N 3  \
   --use_norm_grads 1 --norm_grads_weight 0.9 --ip "127.1.5.$count"  > $model_dir/$task/$dataset/train.log 2>&1  &
    wait
    nohup python render.py --eval  -s $data_dir/$dataset -m $model_dir/3/$task/$dataset --use_norm_mlp 1  --use_cosine 1  --use_specular 1 > $model_dir/$task/$dataset/render.log 2>&1  &
    wait
    nohup python metrics.py -m $model_dir/3/$task/$dataset > $model_dir/3/$task/$dataset/metrics.log 2>&1 &
    wait
done


