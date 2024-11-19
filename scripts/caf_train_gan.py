"""
Train a Constant Acceleration Flow Acceleration + GAN
"""

import argparse
from flow.image_datasets import load_data_npy
from flow.script_util import (
    model_and_flow_defaults,
    caf_gan_defaults,
    create_model_and_flow,
    args_to_dict,
    add_dict_to_argparser,
)
from flow.train_util import CAFTrainLoopGAN
from flow.nn import cycle
from torchvision import datasets, transforms
import torch
import flow.enc_dec_lib as enc_dec_lib


def main():
    args = create_argparser().parse_args()

    model, diffusion = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )

    velmodel, _ = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )
    
    discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(args)

    model.train()
    velmodel.eval()

    # Load Real data
    if args.data_name.lower() == 'cifar10':
        transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        dataset = datasets.CIFAR10(args.real_data_dir, train=True, download=True, transform = transform)
        
        data_cllt = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    elif args.data_name.lower() == 'imagenet':
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        train_data = datasets.ImageFolder(args.real_data_dir, transform = transform)
        data_cllt = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    else:
        raise ValueError(f"Unsupported data name: {args.data_name}")

    # Load deterministic coupling gamma
    fake_cllt = load_data_npy(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        deterministic=False,
        class_cond=args.class_cond,
    )

    CAFTrainLoopGAN(
        model=model,
        diffusion=diffusion,
        data=data_cllt,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        d_lr=args.d_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        resume=args.resume,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        is_wandb=args.is_wandb,
        total_training_steps=args.total_training_steps,
        save_pth=args.save_pth,
        data_name=args.data_name,
        eval_batch_size=args.eval_batch_size,
        eval_num_samples=args.eval_num_samples,
        ref_path=args.ref_path,
        acc_pth=args.acc_pth,
        velmodel=velmodel,
        loss_norm=args.loss_norm,
        discriminator=discriminator,
        discriminator_feature_extractor=discriminator_feature_extractor,
        fake_cllt=fake_cllt,
        g_learning_period=args.g_learning_period,
        image_size=args.image_size,
        class_cond=args.class_cond,
        apply_adaptive_weight=args.apply_adaptive_weight,
        schedule_sampler=args.schedule_sampler,
        num_classes=args.num_classes,
    ).run_loop()

def create_argparser():
    defaults = dict(data_name='cifar10')
    #defaults = dict(data_name='imagenet')
    defaults.update(model_and_flow_defaults(defaults['data_name']))
    defaults.update(caf_gan_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
