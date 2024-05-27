"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.ssim_psnr import calculate_psnr, calculate_ssim
from util.unsupervised_metric import calculate_CR_CNR
import numpy as np

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = 'test'
    weights_paths = opt.weights_path.split(',')
    model = create_model(opt)      # create a model given opt.model and other options
    psnr_tot = []
    ssim_tot = []
    CR_tot = []
    CNR_tot = []
    for fold in range(5):
        dataset = create_dataset(opt, fold)  # create a dataset given opt.dataset_mode and other options
        model.setup(opt, weights_paths[fold])               # regular setup: load and print networks; create schedulers
        model.eval()
        # initialize logger
        # if opt.use_wandb:
        #     wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        #     wandb_run._label(repo='CycleGAN-and-pix2pix')

        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Weights = %s' % (opt.name, opt.phase, weights_paths[fold]))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        psnr, ssim, cr, cnr = 0., 0., 0., 0.
        for i, (data, label) in enumerate(dataset):
            model.set_input(data, label)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s, %s' % (i, img_path, data['B_paths']))
                
            if opt.data_type == 'all':
                data_root = f'{opt.dataroot}'
            else:
                data_root = f'{opt.dataroot}_{opt.data_type}'
            loaded_data = {"TrainA": [0.5, 0.5], "TrainB": [0.5, 0.5]}
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb, loaded_data = loaded_data['TrainB'])

            fake_B, real_B = model.validate()
            fake_B = (fake_B + 1.) / 2  # denorm
            real_B = (real_B + 1.) / 2
            fake_B = fake_B.squeeze(0).squeeze(0).detach().cpu().numpy()
            real_B = real_B.squeeze(0).squeeze(0).detach().cpu().numpy()
            fake_B = np.clip((fake_B * 255.0), 0., 255.)
            real_B = np.clip((real_B * 255.0), 0., 255.)
            # print(calculate_psnr(fake_B, real_B, 0), calculate_ssim(fake_B, real_B, 0))
            psnr += calculate_psnr(fake_B, real_B, 0)
            ssim += calculate_ssim(fake_B, real_B, 0)
            tmp_cr, tmp_cnr = calculate_CR_CNR(fake_B)
            if tmp_cr < 0.:
                print(tmp_cr)
            cr += tmp_cr
            cnr += tmp_cnr
            
        psnr_tot.append(psnr / len(dataset))
        ssim_tot.append(ssim / len(dataset))
        CR_tot.append(cr / len(dataset))
        CNR_tot.append(cnr / len(dataset))
        webpage.save()  # save the HTML

    print(psnr_tot)
    print(ssim_tot)
    print('PSNR mean: ', np.mean(np.array(psnr_tot)))
    print('PSNR std: ', np.std(np.array(psnr_tot)))
    print('SSIM mean: ', np.mean(np.array(ssim_tot)))
    print('SSIM std: ', np.std(np.array(ssim_tot)))
    print('CR mean: ', np.mean(np.array(CR_tot)))
    print('CR std: ', np.std(np.array(CR_tot)))
    print('CNR mean: ', np.mean(np.array(CNR_tot)))
    print('CNR std: ', np.std(np.array(CNR_tot)))