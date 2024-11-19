import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import torch.nn as nn
from pg_modules.diffaug import DiffAugment

def load_feature_extractor(args, eval=True):
    feature_extractor = None
    if args.loss_norm == 'lpips':
        from piq import LPIPS
        feature_extractor = LPIPS(replace_pooling=True, reduction="none")
    elif args.loss_norm == 'cnn_vit':
        from pg_modules.projector import F_RandomProj
        backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        feature_extractor = []
        backbone_kwargs = {'im_res': args.image_size}
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            feature_extractor.append([bb_name, feat])
        feature_extractor = nn.ModuleDict(feature_extractor)
        feature_extractor = feature_extractor.train(False).to(dist_util.dev())
        feature_extractor.requires_grad_(False)
    return feature_extractor

def load_discriminator_and_d_feature_extractor(args):
    #assert (args.gan_training == True) == (args.d_architecture == 'StyleGAN-XL')
    from pg_modules.projector import F_RandomProj
    from pg_modules.discriminator import MultiScaleD
    backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
    discriminator, discriminator_feature_extractor = [], []

    backbone_kwargs = {'im_res': args.image_size}
    for i, bb_name in enumerate(backbones):
        feat = F_RandomProj(bb_name, **backbone_kwargs)
        disc = MultiScaleD(
            channels=feat.CHANNELS,
            resolutions=feat.RESOLUTIONS,
            **backbone_kwargs,
        )
        discriminator_feature_extractor.append([bb_name, feat])
        discriminator.append([bb_name, disc])

    discriminator_feature_extractor = nn.ModuleDict(discriminator_feature_extractor)
    discriminator_feature_extractor = discriminator_feature_extractor.train(False)
    discriminator_feature_extractor.requires_grad_(False)

    discriminator = nn.ModuleDict(discriminator)
    discriminator.train()
    return discriminator, discriminator_feature_extractor

def get_feature(input, feat, brightness, saturation, contrast, translation_x, translation_y,
                               offset_x, offset_y, name, step):
    # augment input
    input_aug_ = input
    if brightness.shape[0] > 0:
        input_aug_ = DiffAugment(input[:brightness.shape[0]], brightness, saturation, contrast, translation_x, translation_y,
                                   offset_x, offset_y, policy='color,translation,cutout')
        input_aug_ = torch.cat((input_aug_, input[brightness.shape[0]:]))
    # transform to [0,1]
    input_aug = input_aug_.add(1).div(2)
    # apply F-specific normalization
    input_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(input_aug)
    # upsample if smaller, downsample if larger + VIT
    if input.shape[-2] < 256:
        input_n = F.interpolate(input_n, 224, mode='bilinear', align_corners=False)
    # forward pass
    input_features = feat(input_n)
    return input_features

def get_xl_feature(estimate, target=None, feature_extractor=None, discriminator=None, step=-1, **model_kwargs):
    logits_fake, logits_real = [], []
    estimate_features, target_features = [], []
    prob_aug = 1.
    shift_ratio=0.125
    cutout_ratio=0.2
    for bb_name, feat in feature_extractor.items():
        # apply augmentation (x in [-1, 1])
        brightness = (torch.rand(int(estimate.size(0) * prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) - 0.5)
        # brightness = 0.
        saturation = (torch.rand(int(estimate.size(0) * prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) * 2)
        # saturation = 0.
        contrast = (torch.rand(int(estimate.size(0) * prob_aug), 1, 1, 1, dtype=estimate.dtype,
                            device=estimate.device) + 0.5)
        # contrast = 0.
        shift_x, shift_y = int(estimate.size(2) * shift_ratio + 0.5), int(
            estimate.size(3) * shift_ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[int(estimate.size(0) * prob_aug), 1, 1],
                                   device=estimate.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[int(estimate.size(0) * prob_aug), 1, 1],
                                   device=estimate.device)
        cutout_size = int(estimate.size(2) * cutout_ratio + 0.5), int(
            estimate.size(3) * cutout_ratio + 0.5)
        offset_x = torch.randint(0, estimate.size(2) + (1 - cutout_size[0] % 2),
                              size=[int(estimate.size(0) * prob_aug), 1, 1], device=estimate.device)
        offset_y = torch.randint(0, estimate.size(3) + (1 - cutout_size[1] % 2),
                              size=[int(estimate.size(0) * prob_aug), 1, 1], device=estimate.device)

        estimate_feature = get_feature(estimate, feat, brightness, saturation, contrast,
                                            translation_x, translation_y, offset_x, offset_y, 'estimate', step)
        estimate_features.append(estimate_feature)
        if discriminator is not None:
            try:
                logits_fake += discriminator.module[bb_name](estimate_feature, model_kwargs)
            except:
                logits_fake += discriminator[bb_name](estimate_feature, model_kwargs)

        if target != None:
            with torch.no_grad():
                target_feature = get_feature(target, feat, brightness, saturation, contrast,
                                                  translation_x, translation_y, offset_x, offset_y, 'target', step)
                target_features.append(target_feature)
            if discriminator is not None:
                try:
                    logits_real += discriminator.module[bb_name](target_feature, model_kwargs)
                except:
                    logits_real += discriminator[bb_name](target_feature, model_kwargs)

    if discriminator is not None:
        if target == None:
            return logits_fake
        else:
            return logits_fake, logits_real
    else:
        if target == None:
            return estimate_features
        else:
            return estimate_features, target_features