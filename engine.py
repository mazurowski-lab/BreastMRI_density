import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F 
from torch.cuda.amp import GradScaler, autocast
import json
import os
import nibabel as nib
import nrrd
import torchio as tio

import numpy as np
from tqdm import tqdm
from utils.ops import aug_rand, rot_rand, aug_distortion, aug_rotflip, aug_byol_distort_3d
from utils.utils import DiceLoss, dice_coeff



def eval_volume(model, dataloader, size, device, epoch=0, x_iter=8, y_iter=8, z_iter=3, save_dir=None):
    # save dir -- when specify, predict masks for unlabeled images
    model.eval()
    dc_list = []



    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = batch['image'].float().to(device)
        split_file = 'buffer/Item_Features.json'
        with open(split_file, 'r') as f:

            split_set = json.load(f)
            index = split_set['path'].index(batch['name'][0])
            ISFlip = split_set['ISFlip'][index]
            XYFlip = split_set['XYFlip'][index]
            padding_size = split_set['Padded'][index]
            xy_size = split_set['xy_size'][index]
            z_size = split_set['z_size'][index]

        if save_dir is None:
            mask = batch['mask'].to(device)
            mask = F.one_hot(mask[:, 0].long(), -1).permute(0, 4, 1, 2, 3)
            mask = mask.float()

            pred_mask = torch.zeros(mask.shape).float().to(device)
            pred_count = torch.zeros(mask.shape).float().to(device)
        else:
            image_shape = list(image.shape)
            # hardcode output dim for now
            image_shape[1] = 3
            pred_mask = torch.zeros(image_shape).float().to(device)
            pred_count = torch.zeros(image_shape).float().to(device)
            name = batch['name'][0]

        # print(image.shape)

        x_step = (image.shape[2] - size) // (x_iter - 1)
        y_step = (image.shape[3] - size) // (y_iter - 1)
        z_step = (image.shape[4] - size) // (z_iter - 1)

        with torch.no_grad():
            for x in range(x_iter):
                for y in range(y_iter):
                    for z in range(z_iter):
                        curr_patch = image[:, :, x * x_step:x * x_step + size,
                                     y * y_step:y * y_step + size,
                                     z * z_step:z * z_step + size]
                        # Need to call module.forward since the other process is waiting
                        curr_pred = model.module.forward(curr_patch)
                        curr_pred = torch.softmax(curr_pred, dim=1)

                        pred_mask[:, :, x * x_step:x * x_step + size,
                        y * y_step:y * y_step + size,
                        z * z_step:z * z_step + size] += curr_pred
                        pred_count[:, :, x * x_step:x * x_step + size,
                        y * y_step:y * y_step + size,
                        z * z_step:z * z_step + size] += 1
        # Average prediction over volumes
        pred_mask = torch.div(pred_mask, pred_count)
        pred_binary = (pred_mask > 0.5).float()

        if save_dir is None:
            tmp = []
            for class_idx in range(pred_binary.shape[1]):
                dc = dice_coeff(pred_binary[:, class_idx], mask[:, class_idx], device)
                tmp.append(dc.cpu().item())

            dc_list.append(tmp)
        else:
            print('Generate mask', batch['name'][0])

            # Save prediction
            pred_numpy = pred_binary.cpu().numpy()
            # convert prediction format from n_channel*size to size where
            # class 1 -> vessel
            # class 2 -> tissue
            pred_numpy = pred_numpy[:, 1] + pred_numpy[:, 2] * 2
            pred_numpy = np.uint8(pred_numpy.squeeze())

        header_file = 'buffer/HeaderList.json'
        with open(header_file, 'r') as f:

            header_set = json.load(f)
            header = header_set[index]

        if header["space directions"][0][0] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=1)
            header["space directions"][0][0] = abs(header["space directions"][0][0])
            header["space origin"][0] = header["space origin"][0] - header["space directions"][0][0] * xy_size


        if header["space directions"][1][1] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=0)
            header["space directions"][1][1] = abs(header["space directions"][1][1])
            header["space origin"][1] = header["space origin"][1] - header["space directions"][1][1] * xy_size

        if header["space directions"][2][2] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=2)
            header["space directions"][2][2] = abs(header["space directions"][2][2])
            header["space origin"][2] = header["space origin"][2] - header["space directions"][2][2] * z_size

        '''
        if header["space directions"][1][1] < 0:
            origin = header['space origin']
            xspacing = header['space directions'][0][0]
            origin[0] = origin[0] - xspacing * xy_size
            header['space origin'] = origin

        if header["space directions"][0][0] < 0:
            origin = header['space origin']
            yspacing = header['space directions'][1][1]
            origin[1] = origin[1] - yspacing * xy_size
            header['space origin'] = origin

        if header["space directions"][2][2] < 0:
            origin = header['space origin']
            zspacing = header['space directions'][2][2]
            origin[2] = origin[2] - zspacing * z_size
            header['space origin'] = origin

        for i in range(3):
            header['space directions'][i][i] = abs(header['space directions'][i][i])
        '''

        '''
        if ISFlip == True:
            pred_numpy = np.flip(pred_numpy, axis=2)
            origin = header['space origin']
            zspacing = header['space directions'][2][2]
            origin[2] = origin[2] - zspacing * z_size
            header['space origin'] = origin
        '''

        if padding_size > 0:
            original_depth = size - padding_size
            padding_before = padding_size // 2
            # Calculate the starting and ending indices to slice
            start_idx = padding_before
            end_idx = start_idx + original_depth
            pred_numpy = pred_numpy[:, :, start_idx:end_idx]

        nrrd.write(os.path.join(save_dir, batch['name'][0][:-4] + '.seg.nrrd'), pred_numpy, header)


def eval_full_volume(model, dataloader, device, size, save_dir, epoch=0):
    model.eval()
    dc_list = []

    for idx, batch in tqdm(enumerate(dataloader)):
        image = batch['image'].float().to(device)

        split_file = 'buffer/Item_Features.json'
        with open(split_file, 'r') as f:
            split_set = json.load(f)
            index = split_set['path'].index(batch['name'][0])
            ISFlip = split_set['ISFlip'][index]
            XYFlip = split_set['XYFlip'][index]
            padding_size = split_set['Padded'][index]
            xy_size = split_set['xy_size'][index]
            z_size = split_set['z_size'][index]

        prediction = model.module.forward(image)
        pred_mask = torch.softmax(prediction, dim=1)
        pred_binary = (pred_mask > 0.5).float()


        print('Generate mask', batch['name'][0])

        pred_numpy = pred_binary.cpu().numpy()
        # convert prediction format from n_channel*size to size where
        # class 1 -> vessel
        # class 2 -> tissue
        pred_numpy = pred_numpy[:, 1]
        pred_tensor = torch.from_numpy(pred_numpy).float()
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=pred_tensor))

        resize_transform_back = tio.Compose([tio.Resize((xy_size, xy_size, z_size)), ])
        transformed_subject = resize_transform_back(subject)
        transformed_pred_tensor = transformed_subject.image[tio.DATA].float()
        pred_numpy = transformed_pred_tensor.cpu().numpy()

        pred_numpy = np.uint8(pred_numpy.squeeze())


        header_file = 'buffer/HeaderList.json'
        with open(header_file, 'r') as f:

            header_set = json.load(f)
            header = header_set[index]

        if header["space directions"][0][0] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=1)
            header["space directions"][0][0] = abs(header["space directions"][0][0])
            header["space origin"][0] = header["space origin"][0] - header["space directions"][0][0] * xy_size


        if header["space directions"][1][1] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=0)
            header["space directions"][1][1] = abs(header["space directions"][1][1])
            header["space origin"][1] = header["space origin"][1] - header["space directions"][1][1] * xy_size

        if header["space directions"][2][2] < 0:
            #pred_numpy = np.flip(pred_numpy, axis=2)
            header["space directions"][2][2] = abs(header["space directions"][2][2])
            header["space origin"][2] = header["space origin"][2] - header["space directions"][2][2] * z_size

        '''
        if ISFlip == True:
            pred_numpy = np.flip(pred_numpy, axis=2)
            origin = header['space origin']
            zspacing = header['space directions'][2][2]
            origin[2] = origin[2] - zspacing * z_size
            header['space origin'] = origin
        '''

        if padding_size > 0:
            original_depth = size - padding_size
            padding_before = padding_size // 2
            # Calculate the starting and ending indices to slice
            start_idx = padding_before
            end_idx = start_idx + original_depth
            pred_numpy = pred_numpy[:, :, start_idx:end_idx]

        nrrd.write(os.path.join(save_dir, batch['name'][0][:-4] + '.seg.nrrd'), pred_numpy, header)
