# NieR: Normal-Based Lighting Scene Rendering

> This repository contains the official author implementation related to the paper "NieR: > Normal-Based Lighting Scene Rendering" and can be found here. We further provide comparison images and videos, as well as recently created pre-trained models. The relevant resources can be viewed and downloaded by visiting our [HomePage](http://124.70.164.141:8085/).
>
> *This project significantly improves the detail and quality of gaussian splatting scene reconstructions.



## Abstract
*In real-world road scenes, diverse material properties lead to complex light reflection phenomena, making accurate color reproduction crucial for enhancing the realism and safety of simulated driving environments. However, existing methods often struggle to capture the full spectrum of lighting effects, particularly in dynamic scenarios where viewpoint changes induce significant material color variations. To address this challenge, we introduce NieR (Normal-Based Lighting Scene Rendering), a novel framework that takes into account the nuances of light reflection on diverse material surfaces, leading to more precise rendering. To simulate the lighting synthesis process, we present the LD (Light Decomposition) module, which captures the lighting reflection characteristics on surfaces. Furthermore, to address dynamic lighting scenes, we propose the HNGD (Hierarchical Normal Gradient Densification) module to overcome the limitations of sparse Gaussian representation. Specifically, we dynamically adjust the Gaussian density based on normal gradients. Experimental evaluations demonstrate that our method outperforms state-of-the-art (SOTA) methods in terms of visual quality and exhibits significant advantages in performance indicators. Codes are available in the appendix.*



![](./assets/fig_cmp_low.png)








## dataset
you can get the dataset we used in paper form mip-nerf360 and tanks&temples:
> [gaussian splatting](hhttps://github.com/graphdeco-inria/gaussian-splatting)

> [mipnerf360](https://jonbarron.info/mipnerf360/)

## Pretrained Models
you can download our pretrained models at our [HomePage](http://124.70.164.141:8085/).

## Running

### environment
we use the same environment as gaussian splatting: [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### training
We provide training and rendering control parameters to control the enabling of the module --use_norm_mlp: whether to use the LD module use_cosine: whether to use the diffuse intensity coefficient. Refer to the code parameter details for specific parameter usage
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
We provide execution scripts for multi-scene training rendering evaluations to help developers use project
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
