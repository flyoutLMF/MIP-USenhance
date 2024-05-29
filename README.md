# USenhance23_POAS

## USenhance23 Challenge: https://ultrasoundenhance2023.grand-challenge.org/

Ultrasound Image Enhancement Challenge 2023  
Team POAS (POSTECH, ASAN Medical Center)

## Dataset Preparation

`low2high`: Original Dataset (breast, carotid, kidney, liver, thyroid)  
|--`trainA`: Low Resolution Train Dataset   
|--`trainB`: High Resolution Train Dataset  
|--`testA`: Low Resolution Test Dataset  
|--`testB`: Same to testA
'label_inform.json': Label class information

## Train/Test Phase

`sample_base.sh`: Sample Training Code



```shell
#sample base
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataroot ../train_datasets \
--name miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--batch_size 8 \
--save_epoch_freq 50 \
--val_epoch_freq 50 \
--checkpoints_dir '../checkpoints' \
--phase train \
--display_id 0 \
--is_mtl \                  # Use generator classifier
--is_mtl_D                  # Use Discriminator classifier

```



## Checkpoint

Checkpoints for abalation have been uploaded to [GoogleDrive](https://drive.google.com/drive/folders/1rCnvnXaw7Mx3Fg1HGvd7hZS9y3u4bPZa?usp=sharing)

```shell
Medical Image Processing
|--cyclegan_base 
|--cyclegan_D+G # With both Generator & Discriminator classifiers for training
```
