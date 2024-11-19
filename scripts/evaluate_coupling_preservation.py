import argparse
import torch as th
import os
import numpy as np
from PIL import Image
import torchvision
from piq import LPIPS, psnr

from flow.nn import random_seed
from flow.script_util import (
    model_and_flow_defaults,
    caf_eval_defaults,
    create_model_and_flow,
    args_to_dict,
    add_dict_to_argparser,
)

def main():
    random_seed(999)
    args = create_argparser().parse_args()

    model, diffusion = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )

    velmodel, _ = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )
    
   # Load the best model
    try:
        print(f'Loading the best model from {args.save_pth}')
        ckpt = th.load(os.path.join(args.save_pth, 'fidbest.pt'), map_location='cpu')
        model.load_state_dict(ckpt['ema'])
        velmodel.load_state_dict(ckpt['velmodel'])
    except:
        raise ValueError(f'No best model found in {args.save_pth}')

    model.to('cuda')
    velmodel.to('cuda')
    model.eval()
    velmodel.eval()

    print('Evaluating coupling preservation...')
    iters = args.eval_num_samples // args.batch_size
    number = 0
    
    # Load latent
    try:
        print(f'Loading latent from {args.data_dir}')
        z0_list = sorted(os.listdir(os.path.join(args.data_dir,'z0')))
        z0_fname = z0_list[:args.eval_num_samples]
        z0_tensors = []
        z1_tensors = []
        for fname in z0_fname:
            z0_latents = np.load(os.path.join(args.data_dir,'z0',fname))
            z0_latents = th.from_numpy(z0_latents).float()
            z0_tensors.append(z0_latents.unsqueeze(0))
            z1_latents = np.load(os.path.join(args.data_dir,'z1',fname))
            z1_latents = th.from_numpy(z1_latents).float()
            z1_tensors.append(z1_latents.unsqueeze(0))
        z0_latents = th.cat(z0_tensors, dim=0)
        z1_latents = th.cat(z1_tensors, dim=0)
    except:
       raise ValueError(f'No latent file found in {args.data_dir}')

    if args.test_score:
        lpips_measure = LPIPS(reduction = 'none')
        lpips_list, psnr_list = [], []
    
    # Evaluate
    for itr in range(iters):
        if args.class_cond:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device='cuda'
            )
        else:
            classes = None
        
        z0 = z0_latents[number: number+args.batch_size]
        z1 = z1_latents[number: number+args.batch_size]
        z0 = z0.to('cuda')
        z1 = z1.clamp(-1., 1.)

        with th.inference_mode():
            sample = diffusion.sample(N=args.sample_step, model=model, velmodel=velmodel, latents=z0, classes=classes)
            sample = sample.clamp(-1., 1.)
            if args.test_score:
                for idx in range(z1.shape[0]):
                    real_data_test = (z1[idx].unsqueeze(0) + 1)/2
                    fake_data_test = (sample[idx].unsqueeze(0) + 1)/2
                    
                    psnr_score = psnr(real_data_test.cpu(), fake_data_test.cpu(), data_range=1.)
                    lpips_score = lpips_measure(real_data_test.cpu(), fake_data_test.cpu())
                    psnr_list.append(psnr_score.item())
                    lpips_list.append(lpips_score.item())
            sample_z = (sample + 1)/2

        if args.save_z0_image:
            sample_dir_z0 = os.path.join(args.save_pth, 'coupling_preservation')
            os.makedirs(sample_dir_z0, exist_ok=True)
            for k in range(sample.shape[0]):
                fake_img = sample_z[k].data.cpu().numpy().transpose(1,2,0)
                fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                fake_img.save(os.path.join(sample_dir_z0, 'Estimated-z1-N{}-{}-{}.png'.format(args.sample_step, itr, k)))
        
        if args.save_z1_image:
            real_data = (z1 + 1)/2
            for k in range(sample.shape[0]):
                fake_img = real_data[k].data.cpu().numpy().transpose(1,2,0)
                fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                fake_img.save(os.path.join(sample_dir_z0, 'GT-z1-N{}-{}-{}.png'.format(args.sample_step, itr, k)))
        
        number += sample.shape[0]
        print(f'Processed {number} samples')
    
    if args.test_score:
        lpips_avg = np.mean(lpips_list)
        psnr_avg = np.mean(psnr_list)
        print('lpips score:', lpips_avg)
        print('psnr score:', psnr_avg)
    
    print(f'Saved {number} samples to {args.save_pth}')


def create_argparser():
    defaults = dict(
        data_name='cifar10',
        test_score=True,
        save_z0_image=True,
        save_z1_image=True,
        sample_step=1,
        )
    defaults.update(model_and_flow_defaults(defaults['data_name']))
    defaults.update(caf_eval_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
