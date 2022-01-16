The code for rendering 3d objects in this repo is from the [Strike (With) A Pose](https://github.com/airalcorn2/strike-with-a-pose) repo.
# ImagePose
Are vision models robust against uncommon poses?


| Model         | Source | Model Name| Dataset | Params | IN acc|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Resnet50  | torchvision | ResNet50 | ImageNet(1M) | 25M | 79.3% |
| Resnet152 | torchvision | ResNet152 | ImageNet(1M) | 43M | 80.1% |
| Resnet101 | torchvision | ResNet101 | ImageNet(1M) | 59M | 80.1% |
| | | | | | |
| Clip_vit_B16 | clip | CLIP_ViT_B_16 | WebImageText(400M) | 86M | 63.2% |
| Clip_50      | clip | CLIP_ResNet50 | WebImageText(400M) | 25M  | 62.2% |
| Clip_101     | clip | CLIP_ResNet101 | WebImageText(400M) | 59M | 59.6% |
| | | | | | |
| ViT_L16 | model-vs-human | ViT_L_patch16_224 | ImageNet(1M) | 307M | xx
| ViT_B16 | model-vs-human | ViT_B_patch16_224 | ImageNet(1M) | 86M | xx
| ViT_S16 | model-vs-human | ViT_S_patch16_224 | ImageNet(1M) | 22M | xx
| ViT_B16_sam | timm | ViT_B_patch16_sam_224 | ImageNet(1M) | 86M | xx
| | | | | | |
|DINO_S16 | torchhub | DINO_ViT_small_16 | xx | xx | xx
|DINO_B16 | torchhub | DINO_ViT_base_16 | xx | xx | xx
|DINO_RN50 | torchhub | DINO_ResNet50 | xx | xx | xx
| | | | | | |
| Simclr | model-vs-human | SimCLR_ResNet50 | ImageNet(1M) | 25M | 69.3% |
| Moco | model-vs-human | MOCO_ResNet50 | ImageNet(1M) | 25M | 71.1% | 
| | | | | | |
| BiTM_50  | model-vs-human | BiTM_resnetv2_50x1 | ImageNet21k(14M) | 25M | 80.0% |
| BiTM_101 | model-vs-human | BiTM_resnetv2_101x1 | ImageNet21k(14M) | 43M | 82.5% |
| BiTM_152x2 | model-vs-human | BiTM_resnetv2_152x2 | ImageNet21k(14M) | 98M | 85.5% |
| | | | | | |
| SWSL_ResNet50  | model-vs-human | ResNet50_swsl | (64M) | 25M | 79.1% |
| SWSL_ResNeXt101 | model-vs-human | ResNeXt101_32x16d_swsl | (64M) | 193M | 81.2% |
| | | | | | |
| Mixer_S16 | timm | Mixer_S16_224 | ImageNet(1M) | 10M |
| Mixer_B16 | timm | Mixer_B16_224 | ImageNet(1M) | 46M |
| Mixer_L16 | timm | Mixer_L16_224 | ImageNet(1M) | 189M |
| | | | | | |
| Beit_B16 | timm | Beit_B16_224 | ImageNet(1M) | 86M |
| Beit_L16 | timm | Beit_L16_224 | ImageNet(1M) | 307M |
| | | | | | |
| Deit_B16 | timm | Deit_B16_224 | ImageNet(1M) | 86M |
| Deit_S16 | timm | Deit_L16_224 | ImageNet(1M) | 22M |



