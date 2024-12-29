import argparse
import torch as th
import os
import numpy as np
from PIL import Image

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
        ckpt = th.load(os.path.join(args.save_pth, 'model.pt'), map_location='cpu')
        if 'ema' in ckpt.keys():
            model.load_state_dict(ckpt['ema'])
        else:
            model.load_state_dict(ckpt['model'])
        velmodel.load_state_dict(ckpt['velmodel'])
    except:
        raise ValueError(f'No best model found in {args.save_pth}')

    model.to('cuda')
    velmodel.to('cuda')
    model.eval()
    velmodel.eval()

    print(f'Sampling {args.eval_num_samples} samples...')
    iters = args.eval_num_samples // args.batch_size
    number = 0
    
    # Sample
    for itr in range(iters):
        if args.class_cond:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device='cuda',
            )
        else:
            classes = None
        
        # Sample from prior
        z = th.randn((args.batch_size, 3, args.image_size, args.image_size)).to('cuda')

        with th.inference_mode():
            sample = diffusion.sample(N=args.sample_step, model=model, velmodel=velmodel, latents=z, classes=classes)
            sample = sample.clamp(-1., 1.)
            sample_z = (sample + 1)/2

        if args.save_image:
            sample_dir_png = os.path.join(args.save_pth, 'N{}_image/{}_image'.format(args.sample_step, itr))
            os.makedirs(sample_dir_png, exist_ok=True)
            for k in range(sample.shape[0]):
                fake_img = sample_z[k].data.cpu().numpy().transpose(1,2,0)
                fake_img = Image.fromarray((fake_img * 255).astype(np.uint8))
                fake_img.save(os.path.join(sample_dir_png, 'gen-{}-{}.png'.format(itr, k)))
        
        number += sample.shape[0]
    print(f'Saved {number} samples to {args.save_pth}')

def create_argparser():
    defaults = dict(
        data_name='cifar10',
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
