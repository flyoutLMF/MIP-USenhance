"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.ssim_psnr import calculate_psnr, calculate_ssim
from util.unsupervised_metric import calculate_CR_CNR
from copy import deepcopy
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    val_opt = deepcopy(opt)
    val_opt.phase = 'test'
    val_opt.batch_size = 1
    val_opt.serial_batches = True
    for fold in range(5):
        print('='*20 + f'\nTrain for Fold {str(fold+1)}\n' + '='*20)
        dataset = create_dataset(opt, fold)  # create a dataset given opt.dataset_mode and other options
        val_dataset = create_dataset(val_opt, fold)
        dataset_size = len(dataset)    # get the number of images in the dataset.
        val_dataset_size = len(val_dataset)
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)      # create a model given opt.model and other options
        model.train()
        model.setup_train(opt)               # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        total_iters = 0                # the total number of training iterations

        best_psnr = 0.
        best_ssim = 0.

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            for i, (data, label) in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += 1
                epoch_iter += opt.batch_size
                model.set_input(data, label)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                #     model.save_networks(save_suffix)
                
                iter_data_time = time.time()
                
            model.update_learning_rate()    # update learning rates in the beginning of every epoch.
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks(f'Fold{str(fold+1)}_latest')
                model.save_networks(f'Fold{str(fold+1)}_{str(epoch)}')

            if epoch % opt.val_epoch_freq == 0:
                model.eval()
                psnr, ssim, cr, cnr = 0., 0., 0., 0.
                for i, (data, label) in enumerate(val_dataset):
                    model.set_input(data, label)
                    fake_B, real_B = model.validate()
                    fake_B = (fake_B + 1.) / 2  # denorm
                    real_B = (real_B + 1.) / 2
                    fake_B = fake_B.squeeze(0).squeeze(0).detach().cpu().numpy()
                    real_B = real_B.squeeze(0).squeeze(0).detach().cpu().numpy()
                    fake_B = np.clip((fake_B * 255.0), 0., 255.)
                    real_B = np.clip((real_B * 255.0), 0., 255.)
                    psnr += calculate_psnr(fake_B, real_B, 0)
                    ssim += calculate_ssim(fake_B, real_B, 0)
                    tmp_cr, tmp_cnr = calculate_CR_CNR(fake_B)
                    cr += tmp_cr
                    cnr += tmp_cnr

                model.train()
                psnr = psnr / val_dataset_size
                ssim = ssim / val_dataset_size
                cr = cr / val_dataset_size
                cnr = cnr / val_dataset_size
                if psnr > best_psnr:
                    best_psnr = psnr
                    p = os.path.join(opt.checkpoints_dir, opt.name)
                    for n in os.listdir(p):
                        if n.find(f'Fold{str(fold+1)}_best_psnr') >= 0:
                            os.remove(os.path.join(p, n))
                    model.save_networks(f'Fold{str(fold+1)}_best_psnr_%d_%.2f' % (epoch, psnr))
                if ssim > best_ssim:
                    best_ssim = ssim
                    p = os.path.join(opt.checkpoints_dir, opt.name)
                    for n in os.listdir(p):
                        if n.find(f'Fold{str(fold+1)}_best_ssim') >= 0:
                            os.remove(os.path.join(p, n))
                    model.save_networks(f'Fold{str(fold+1)}_best_ssim_%d_%.3f' % (epoch, ssim))

                print('End of epoch %d / %d \t Time Taken: %d sec. Val: %.3f, %.3f, %.3f, %.3f' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time, psnr, ssim, cr, cnr))

            else:
                print('End of epoch %d / %d \t Time Taken: %d sec.' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))



