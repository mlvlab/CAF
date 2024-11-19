"""
Train a Constant Acceleration Flow Acceleration model
"""

import argparse
from flow.image_datasets import load_data_npy
from flow.script_util import (
    model_and_flow_defaults,
    caf_acc_defaults,
    create_model_and_flow,
    args_to_dict,
    add_dict_to_argparser,
)
from flow.train_util import CAFTrainLoopAcc

def main():
    args = create_argparser().parse_args()
    print(args)
    model, diffusion = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )
    velmodel, _ = create_model_and_flow(
        **args_to_dict(args, model_and_flow_defaults(args.data_name).keys())
    )
    
    data_cllt = load_data_npy(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            deterministic=False,
            class_cond=args.class_cond,
    )
    
    CAFTrainLoopAcc(
        model=model,
        diffusion=diffusion,
        data=data_cllt,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
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
        vel_pth=args.vel_pth,
        velmodel=velmodel,
        image_size=args.image_size,
        class_cond=args.class_cond,
        loss_norm=args.loss_norm,
        schedule_sampler=args.schedule_sampler,
        num_classes=args.num_classes,
    ).run_loop()

def create_argparser():
    defaults = dict(data_name='cifar10')
    #defaults = dict(data_name='imagenet')
    defaults.update(model_and_flow_defaults(defaults['data_name']))
    defaults.update(caf_acc_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
