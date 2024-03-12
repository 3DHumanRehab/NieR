# NieR: Normal-Based Lighting Scene Rendering

![](./assets/fig_cmp_low.png)

## dataset
you can get the dataset form mip-nerf360 and tanks&temples:


[gaussian splatting](hhttps://github.com/graphdeco-inria/gaussian-splatting)

[mipnerf360](https://jonbarron.info/mipnerf360/)
## Running

### environment
we use the same environment as gaussian splatting: [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### training
```
python train.py -s $data_dir -m $model_dir   --use_norm_mlp 1  --use_cosine 1 --use_specular 1 \
   --use_hierarchical 1 --densification_iter 15000 --densify_grad_scaling 1 1.5 2 2.5 3 3.5  \
   --use_hierarchical_split 1 --densify_split_N 3  \
   --use_norm_grads 1 --norm_grads_weight 0.9 
```
### render
```
python render.py --eval -s $data_dir -m $model_dir --use_norm_mlp 1  --use_cosine 1  --use_specular 1 
```
### metric
```
python metrics.py -m $model_dir
```


### scripts
```
bash run_baseline.sh 
bash run_ours.sh
bash run_ablation_ld.sh
bash run_ablation_hngd.sh
```


# Citation
If you use this software package, please cite whichever constituent paper(s) you build upon, or feel free to cite this entire codebase as:

```
@Article{

}
```
