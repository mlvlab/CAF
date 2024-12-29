import argparse

from flow.pipeline_caf import CAFDenoiser
from flow.unet import UNetModel

def caf_vel_defaults(data_name):
    return dict(
        ref_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' if data_name.lower()=='cifar10' \
            else './statistics/VIRTUAL_imagenet64_labeled.npz',
        image_size=32 if data_name.lower()=='cifar10' else 64,
        num_classes=10 if data_name.lower()=='cifar10' else 1000,
        lr=1e-4,
        weight_decay=1e-5,
        data_dir=f"./data/{data_name}_npy/",
        batch_size=160 if data_name.lower()=='cifar10' else 400,
        microbatch=40 if data_name.lower()=='cifar10' else 100,
        ema_rate=0.9999,
        log_interval=500,
        eval_interval=2000,
        save_interval=5000,
        resume=False,
        use_fp16=True,
        alpha=1.5,
        is_wandb=True,
        total_training_steps=1000000,
        save_pth=f'./output/{data_name}_velocity',
        eval_batch_size=50,
        eval_num_samples=50000,
        class_cond=False,
        schedule_sampler='uniform',
        loss_norm='lpips_huber',
        num_timesteps=1000,
    )

def caf_acc_defaults(data_name):
    return dict(
        ref_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' if data_name.lower()=='cifar10' \
            else './statistics/VIRTUAL_imagenet64_labeled.npz',
        image_size=32 if data_name.lower()=='cifar10' else 64,
        num_classes=10 if data_name.lower()=='cifar10' else 1000,
        lr=1e-4,
        weight_decay=1e-5,
        data_dir=f"./data/{data_name}_npy/",
        batch_size=160 if data_name.lower()=='cifar10' else 400,
        microbatch=40 if data_name.lower()=='cifar10' else 100,
        ema_rate=0.9999,
        log_interval=200,
        eval_interval=1000,
        save_interval=5000,
        resume=False,
        use_fp16=True,
        alpha=1.5,
        is_wandb=True,
        total_training_steps=1000000,
        vel_pth=f'./output/{data_name}_velocity',
        save_pth=f'./output/{data_name}_acceleration',
        eval_batch_size=50,
        eval_num_samples=50000,
        class_cond=False,
        schedule_sampler='uniform',
        loss_norm='lpips_huber',
        num_timesteps=1000,
    )

def caf_gan_defaults(data_name):
    return dict(
        # Model
        ref_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' if data_name.lower()=='cifar10' \
            else './statistics/VIRTUAL_imagenet64_labeled.npz',
        image_size=32 if data_name.lower()=='cifar10' else 64,
        num_classes=10 if data_name.lower()=='cifar10' else 1000,
        lr=2e-5,
        d_lr=1e-3,
        weight_decay=1e-5,
        data_dir=f"./data/{data_name}_npy/",
        real_data_dir=f"./data/{data_name}/",
        batch_size=160 if data_name.lower()=='cifar10' else 400,
        microbatch=40 if data_name.lower()=='cifar10' else 100,
        ema_rate=0.9999,
        log_interval=200,
        eval_interval=1000,
        save_interval=2000,
        resume=False,
        use_fp16=True,
        alpha=1.5,
        is_wandb=True,
        total_training_steps=800000,
        acc_pth=f'./output/{data_name}_acceleration',
        save_pth=f'./output/{data_name}_gan',
        eval_batch_size=50,
        eval_num_samples=50000,
        class_cond=True,
        schedule_sampler='uniform',
        loss_norm='huber',
        num_timesteps=10,
        # Training
        apply_adaptive_weight=True,
        g_learning_period=2,
    )


def caf_eval_defaults(data_name):
    return dict(
        # Model
        ref_path = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' if data_name.lower()=='cifar10' \
            else './statistics/VIRTUAL_imagenet64_labeled.npz',
        image_size=32 if data_name.lower()=='cifar10' else 64,
        num_classes=10 if data_name.lower()=='cifar10' else 1000,
        data_dir=f"./data/{data_name}_npy/",
        batch_size=50,
        use_fp16=False,
        alpha=1.5,
        save_pth=f'./output/{data_name}_gan',
        eval_num_samples=1000,
        class_cond=False,
    )


def model_and_flow_defaults(data_name):
    """
    Defaults for image training.
    """
    if data_name.lower()=='cifar10':
        res = dict(
            image_size=32,
            num_channels=192,
            num_res_blocks=3,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=64,
            attention_resolutions="16,8,4",
            channel_mult="",
            dropout=0.1,
            class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            attention_type='xformers',
            num_timesteps=1000,
            alpha=1.5,
            num_classes=10,
            loss_norm='l2',
        )
    else:
        res = dict(
            image_size=64,
            num_channels=192,
            num_res_blocks=3,
            num_heads=4,
            num_heads_upsample=-1,
            num_head_channels=64,
            attention_resolutions="32,16,8",
            channel_mult="",
            dropout=0.1,
            class_cond=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            attention_type='xformers',
            num_timesteps=1000,
            alpha=1.5,
            num_classes=10,
            loss_norm='l2',
        )
    return res


def create_model_and_flow(
    image_size,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    attention_type,
    alpha=1.5,
    num_timesteps=1000,
    num_classes=10,
    loss_norm='l2',
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
        num_classes=num_classes,
    )
    flow = CAFDenoiser(
        alpha=alpha,
        num_timesteps=num_timesteps,
        loss_norm=loss_norm,
    )
    return model, flow

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    attention_type='xformers',
    num_classes=10,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=6,
        model_channels=num_channels,
        out_channels=3,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
