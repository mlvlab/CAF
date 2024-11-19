import argparse
from torchvision import datasets, transforms
import torch as th
import os
import numpy as np
from PIL import Image
from piq import LPIPS, psnr

from flow.nn import random_seed, cycle
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

    # Load Real data
    if args.data_name.lower() == 'cifar10':
        transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        dataset = datasets.CIFAR10('./data/', train=False, download=True, transform = transform)
        
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        data_cllt = cycle(dataloader)
    elif args.data_name.lower() == 'imagenet':
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        dataset = datasets.ImageFolder('./data/imagenet_64/', transform = transform)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        data_cllt = cycle(dataloader)
    else:
        raise ValueError(f"Unsupported data name: {args.data_name}")

    # Evaluate
    eval_num_samples = min(args.eval_num_samples, len(dataset))
    print(f'Evaluating {eval_num_samples} samples')
    iters = eval_num_samples // args.batch_size
    number = 0

    if args.test_score:
        lpips_measure = LPIPS(reduction = 'none')
        lpips_list, psnr_list = [], []
    
    for itr in range(iters):
        data, classes = next(data_cllt)
        data = data * 2 - 1
        data = data.to('cuda')

        with th.inference_mode():
            inverted_z, pred_vel = diffusion.inversion(args.sample_step, model, velmodel, data, classes, return_dict=True)
            sample = diffusion.sample(N=args.sample_step, model=model, velmodel=velmodel, latents=inverted_z, classes=classes, pred_vel=pred_vel)
            sample = sample.clamp(-1., 1.)
            if args.test_score:
                for idx in range(sample.shape[0]):
                    real_data_test = (data[idx].unsqueeze(0) + 1)/2
                    fake_data_test = (sample[idx].unsqueeze(0) + 1)/2
                    
                    psnr_score = psnr(real_data_test.cpu(), fake_data_test.cpu(), data_range=1.)
                    lpips_score = lpips_measure(real_data_test.cpu(), fake_data_test.cpu())
                    psnr_list.append(psnr_score.item())
                    lpips_list.append(lpips_score.item())
            
            sample_z = (sample + 1)/2
            data = (data + 1)/2

        if args.save_image:
            sample_dir_z0 = os.path.join(args.save_pth, 'reconstruction')
            os.makedirs(sample_dir_z0, exist_ok=True)
            for k in range(sample.shape[0]):
                fake_img = sample_z[k].data.cpu().numpy().transpose(1,2,0)
                fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                fake_img.save(os.path.join(sample_dir_z0, 'Reconstructed-N{}-{}-{}.png'.format(args.sample_step, itr, k)))

            for k in range(data.shape[0]):
                real_img = data[k].data.cpu().numpy().transpose(1,2,0)
                real_img = Image.fromarray((real_img * 255).astype(np.uint8))
                real_img.save(os.path.join(sample_dir_z0, 'GT-N{}-{}-{}.png'.format(args.sample_step, itr, k)))
        
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
        save_image=True,
        sample_step=1,
        )
    defaults.update(model_and_flow_defaults(defaults['data_name']))
    defaults.update(caf_eval_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
