import os
import json
import numpy as np

import torch
import torchio as tio
from torch.utils.data import Dataset

from utils.ops import aug_rand, rot_rand, aug_distortion, aug_rotflip, aug_byol_distort_3d

def align_shape(input, target_shape):
    # expected input shape: 1, H, W, D
    _, H, W, D = target_shape
    ret = input[:,:H,:W,:D]
    if ret.shape[1] < H:
        pad = np.zeros([1,H-ret.shape[1],ret.shape[2],ret.shape[3]])
        ret = np.concatenate([ret, pad], axis=1)
    if ret.shape[2] < W:
        pad = np.zeros([1,ret.shape[1],W-ret.shape[2],ret.shape[3]])
        ret = np.concatenate([ret, pad], axis=2)
    if ret.shape[3] < D:
        pad = np.zeros([1,ret.shape[1],ret.shape[2],D-ret.shape[3]])
        ret = np.concatenate([ret, pad], axis=3)
    return ret

def padding(input, volume_size, axis=3):
    pad_num = volume_size - input.shape[axis]
    before_pad = pad_num // 2
    after_pad  = pad_num - before_pad
    
    if axis == 1:
        before = np.zeros([1,before_pad,input.shape[2],input.shape[3]])
        after  = np.zeros([1,after_pad, input.shape[2],input.shape[3]])
        return np.concatenate([before, input, after], axis=1)
    elif axis == 2:
        before = np.zeros([1,input.shape[1], before_pad, input.shape[3]])
        after  = np.zeros([1,input.shape[1], after_pad,  input.shape[3]])
        return np.concatenate([before, input, after], axis=2)
    elif axis == 3:
        before = np.zeros([1,input.shape[1],input.shape[2],before_pad])
        after  = np.zeros([1,input.shape[1],input.shape[2],after_pad])
        return np.concatenate([before, input, after], axis=3)
       

class BreastDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, phase='train', pre_load=False, transforms=None, volume_size=-1, normalize=False, generation=False):
        self.phase = phase
        self.volume_size = volume_size
        self.normalize = normalize
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        # Obtain split set
        self.image_names = []
        split_file = 'buffer/Item_Features.json'
        with open(split_file, 'r') as f:
            split_set = json.load(f)
            for path, split in zip(split_set['path'], split_set['split']):

                if split == phase:
                    self.image_names.append(path)
                if phase == 'generation':
                    self.image_names.append(path)



        # Tmp solution to accelerate learning
        if phase == 'train':
            if self.mask_dir is None:
                self.image_names = self.image_names
            else:
                self.image_names = self.image_names
        #if generation:
            #self.image_names = self.image_names[:50]
        print(self.image_names[0])

        # Define transformations
        self.transforms = transforms
        
        # Load data into memory if specific
        self.pre_load = pre_load
        if pre_load:
            self.images = []
            self.masks  = []
            for f in self.image_names:
                image_path = os.path.join(image_dir, f)
                image = np.expand_dims(np.load(image_path), axis=0)
                original_shape = image.shape
                # Pad with zeros if depth is smaller than volume size
                if image.shape[1] < self.volume_size:
                    image = padding(image, self.volume_size, axis=1)
                if image.shape[2] < self.volume_size:
                    image = padding(image, self.volume_size, axis=2)
                if image.shape[3] < self.volume_size:
                    image = padding(image, self.volume_size, axis=3)

                # Load Mask
                if self.mask_dir is not None:
                    try:
                        mask_path = os.path.join(mask_dir, f)
                        mask = np.expand_dims(np.load(mask_path), axis=0)
                    except:
                        mask_path = os.path.join(mask_dir, f[:-4]+'_DN.npy')
                        mask = np.expand_dims(np.load(mask_path), axis=0)
                        
                        # DN's masks have diff shape, tmp solution to align
                        mask = align_shape(mask, original_shape)

                    # Pad with zeros if depth is smaller than volume size
                    if mask.shape[1] < self.volume_size:
                        mask = padding(mask, self.volume_size, axis=1)
                    if mask.shape[2] < self.volume_size:
                        mask = padding(mask, self.volume_size, axis=2)
                    if mask.shape[3] < self.volume_size:
                        mask = padding(mask, self.volume_size, axis=3)

                else:
                    mask = None

                self.images.append(image)
                self.masks.append(mask)
                
        # Increase iteration size if iterating over volumes
        if volume_size > 0:
            self.repeat_size = 200
            tmp = []
            for n in self.image_names:
                tmp.extend([n]*self.repeat_size)
            self.image_names = tmp

        print('Dataset length', len(self.image_names))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx): 
        name = self.image_names[idx]

        if self.volume_size > 0:
            idx = idx // self.repeat_size

        if self.pre_load:
            image = self.images[idx]
            if self.mask_dir is not None:
                mask = self.masks[idx]
            else:
                mask = None
        else:
            image_path = os.path.join(self.image_dir, name)
            image = np.expand_dims(np.load(image_path), axis=0)
            # Pad with zeros if depth is smaller than volume size
            if image.shape[1] < self.volume_size:
                image = padding(image, self.volume_size, axis=1)
            if image.shape[2] < self.volume_size:
                image = padding(image, self.volume_size, axis=2)
            if image.shape[3] < self.volume_size:
                image = padding(image, self.volume_size, axis=3)
            # Load Mask
            if self.mask_dir is not None:
                mask_path = os.path.join(self.mask_dir, name)
                mask = np.expand_dims(np.load(mask_path), axis=0)
                if mask.shape[1] < self.volume_size:
                    mask = padding(mask, self.volume_size, axis=1)
                if mask.shape[2] < self.volume_size:
                    mask = padding(mask, self.volume_size, axis=2)
                if mask.shape[3] < self.volume_size:
                    mask = padding(mask, self.volume_size, axis=3)
            else:
                mask = None
        
        # TODO: normalize before or after patch
        if self.normalize:
            image_min, image_max = image.min(), image.max()
            image = (image - image.min()) / image.max()

        if self.volume_size > 0:
            if image.shape[1] > self.volume_size:
                rand_x = np.random.randint(image.shape[1] - self.volume_size)
            else:
                rand_x = 0
            if image.shape[2] > self.volume_size:
                rand_y = np.random.randint(image.shape[2] - self.volume_size)
            else:
                rand_y = 0
            if image.shape[3] > self.volume_size:
                rand_z = np.random.randint(image.shape[3] - self.volume_size)
            else:
                rand_z = 0

            image = image[:,rand_x:rand_x+int(self.volume_size),
                            rand_y:rand_y+int(self.volume_size),
                            rand_z:rand_z+int(self.volume_size)]
            if mask is not None:
                mask = mask[:,rand_x:rand_x+int(self.volume_size),
                              rand_y:rand_y+int(self.volume_size),
                              rand_z:rand_z+int(self.volume_size)]
       
        if self.transforms is not None:
            if mask is not None:
                subject = tio.Subject(
                        image = tio.ScalarImage(tensor=image),
                        mask  = tio.LabelMap(tensor=mask))
            else:
                subject = tio.Subject(
                        image = tio.ScalarImage(tensor=image))

            transformed_subject = self.transforms(subject)

            image = transformed_subject.image[tio.DATA].float()
            if mask is not None:
                mask = transformed_subject.mask[tio.DATA].float()

        if mask is None:
            return {'image': image, 'name': name}
        else:
            mask  = mask.transpose(0,3,1,2) 
            return {'image': image, 'mask': mask, 'name': name}
