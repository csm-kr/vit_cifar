## What do the components of transformer do in vision transformer?

The goal of this repo is to find the analysis of various components of ViT (e.g. cls token, pos embedding..)

### Set training environment

- batch : 128
- lr : 1e-3
- epoch : 50

- optimizer : adam
- betas : (0.9, 0.999)
- weight_decay=5e-5

- lr scheduler : cosine scheduler
- loss : cross entropy
- model : Vit  [dim=384, mlp_dim=384, num_heads=12, num_layers=7,
                patch_size=8, image_size=32, is_cls_token=True,
                dropout_ratio=0.0, num_classes=10]
       
          
### 1. Result of ablation study at cifar10  

|Cls token   | Pos embedding     |  Dataset   | Patch size | Length of sequence |  # params      | Accuracy |Test loss | overfitting epoch  |
|------------|-------------------| ---------- | ---------- | ------------------ |----------------|----------|----------| ----------------|
|O           |normal learning 1d |  CIFAR10   | 8 x 8      |  17                |6304906         |75.54     |0.7702    | 35              |
|X           |normal learning 1d |  CIFAR10   | 8 x 8      |  16                |6304138         |36.5      |98        |10.20|
|O           |sinuasoid         |  CIFAR10   | 8 x 8      |  34.3     |53.2     |36.9    |98   |10.20 |
|O           | learning 1d       |  CIFAR10   | 4 x 4      |       | 600 x 600  |**34.7**   |**53.6** |**37.3**|67   |14.85 |
|O           | learning 2d       |  COCOval2017(minival)  | 600 x 600  |**34.7**   |**53.5** |**37.1**|67   |14.85 |
|X           | X                 |  CIFAR10   | 8 x 8      |  16              |52.5     |36.5    |98   |10.20 |
|X           | sinuasoid         |  CIFAR10   | 8 x 8      |  34.3     |53.2     |36.9    |98   |10.20 |
|X           | learning 1d       |  CIFAR10   | 4 x 4      |       | 600 x 600  |**34.7**   |**53.6** |**37.3**|67   |14.85 |
|X           | learning 2d       |  COCOval2017(minival)  | 600 x 600  |**34.7**   |**53.5** |**37.1**|67   |14.85 |
