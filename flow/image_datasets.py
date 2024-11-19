import os
import torch

import numpy as np
from torch.utils.data import DataLoader, Dataset

def random_sq_bbox(img, mask_shape, image_size=32, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w

def set_sq_bbox(img, mask_shape, image_size=32):
    """Generate a fixed sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    
    center_h, center_w = H //2, W // 2
    mask_half_h, mask_half_w = h // 2, w // 2
    
    start_h = center_h - mask_half_h
    start_w = center_w - mask_half_w
    
    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., start_h:start_h+h, start_w:start_w+w] = 0

    return mask

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=32, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                mask_shape=(mask_h, mask_w),
                                image_size=self.image_size,
                                margin=self.margin)
        return mask, t, tl, w, wh

    def _set_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask = set_sq_bbox(img,
                            mask_shape=(mask_h, mask_w),
                            image_size=self.image_size)
        return mask
    
    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask = self._set_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


def load_data_npy(
    data_dir,
    batch_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    dataset = GenNpyDataset(
        image_paths=data_dir,
        random_crop=random_crop,
        random_flip=random_flip,
        class_cond=class_cond,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
        )
    return loader


class GenNpyDataset(Dataset):
    def __init__(self, image_paths, random_crop=False, random_flip=False, class_cond=False):
        super().__init__()
        self.image_paths = image_paths
        self.z0_list = sorted(os.listdir(os.path.join(image_paths,'z0')))
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.class_cond = class_cond
    def __len__(self):
        return len(self.z0_list)
    def __getitem__(self, idx):
        fname = self.z0_list[idx]
        z0 = np.load(os.path.join(self.image_paths,'z0',fname))
        z1 = np.load(os.path.join(self.image_paths,'z1',fname))
        z0 = torch.from_numpy(z0).float()
        z1 = torch.from_numpy(z1).float()
        if self.class_cond:
            classes = np.load(os.path.join(self.image_paths,'class',fname))
            classes = torch.from_numpy(classes)
            return z0, z1, classes
        else:
            #classes = None
            return z0, z1