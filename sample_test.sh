CUDA_VISIBLE_DEVICES=0 \
python test.py \
--results_dir ./test_result \
--name miccai_chall_L2H_pixelSh_MTL_with_pretrained_400 \
--dataset_mode unaligned \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--batch_size 1 \
--phase test \
--dataroot /mnt/zhengxy/data/medical_data/data \
--is_mtl_D \
--is_mtl \
--weights_path /home/lmf/USenhance/checkpoints/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400/Fold1_400,/home/lmf/USenhance/checkpoints/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400/Fold2_400,/home/lmf/USenhance/checkpoints/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400/Fold3_400,/home/lmf/USenhance/checkpoints/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400/Fold4_400,/home/lmf/USenhance/checkpoints/miccai_chall_L2H_pixelSh_MTL_with_pretrained_400/Fold5_400