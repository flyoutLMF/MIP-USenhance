CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataroot ../train_datasets \
--name miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--batch_size 8 \
--phase train \
--display_id 0 \
--is_mtl \
--is_mtl_D