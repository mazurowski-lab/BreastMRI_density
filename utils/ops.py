# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.random import randint

import torch
import torchio as tio

# ---- BEGIN ACTION based aug----
class RandomCrop_AC(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip_AC(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


# class RandomNoise(object):
#     def __init__(self, mu=0, sigma=0.1):
#         self.mu = mu
#         self.sigma = sigma

# ----END ACTION based aug -------

class RandomCrop:
    def __init__(self, scale=(0.8, 1)):
        self.scale = scale

    def __call__(self, sample):
        multiplier = torch.FloatTensor(3).uniform_(*self.scale)
        spatial_shape = torch.Tensor([sample.shape[1:]]).squeeze()
        crop_shape = (spatial_shape * multiplier).round().int()
        sampler = tio.data.UniformSampler(crop_shape)
        print(crop_shape)
        patch = list(sampler(sample, 1))[0]
        return patch


def aug_byol_distort_3d(x, device):
    strong_transforms = tio.Compose([
        # Approximate colorjitter by: Gamma(contrast) + BiasField(brightness)
        tio.RandomGamma(),
        tio.RandomBiasField(),
        tio.RandomFlip((0,1,2)),
        tio.RandomBlur(),
        tio.RandomAffine(scales=(3/4, 4/3),degrees=90,translation=0)
        ])
    weak_transforms = tio.Compose([
        tio.RandomFlip((0,1,2)),
        tio.RandomAffine(scales=(3/4, 4/3),degrees=90,translation=0)
        ])
    
    y1, y2 = [], []
    for i in range(x.shape[0]):
        y1.append(strong_transforms(x[i]).unsqueeze(0))
        y2.append(weak_transforms(x[i]).unsqueeze(0))

    y1 = torch.cat(y1, dim=0)
    y2 = torch.cat(y2, dim=0)
    return y1, y2


def aug_rotflip(x, device):
    x = np.array(x)
    y = []
    for i in range(len(x)):
        xi = x[i]
        orientation = randint(0,4)
        if orientation == 0:
            pass
        else:
            xi = np.rot90(xi, k=orientation, axes=(2,3))
        axis = randint(2,4)
        xi = np.flip(xi, axis=axis)
        y.append(xi)
    y = np.array(y)
    y = torch.tensor(y).float()
    return y


def aug_distortion(x, device):
    first_transform = tio.Compose([
        tio.RandomFlip(axes=(0,1), flip_probability=0.5),
        tio.RandomAffine(
            scales=(0.8, 1.2),
            degrees=30,
            translation=10),
        tio.RandomMotion(degrees=10),
        tio.RandomGhosting(num_ghosts=(4,10)),
        tio.RandomSpike()
    ])

    y1, y2 = [], []
    for i in range(x.shape[0]):
        y1.append(first_transform(x[i]).unsqueeze(0))
        y2.append(first_transform(x[i]).unsqueeze(0))

    y1 = torch.cat(y1, dim=0)
    y2 = torch.cat(y2, dim=0)
    return y1, y2

def patch_rand_drop(device, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=device
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)
    return x


def rot_rand(device, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(device, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(device, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(device, x_aug[i], x_aug[idx_rnd])
    return x_aug


# Helper function
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

