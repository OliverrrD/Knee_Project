from PIL import Image
import cv2
import numpy as np
from Utils.misc import get_logger
import os
import torch.utils.data as data
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image


logger = get_logger(os.path.basename(__file__))


class ChestDataLoader(data.Dataset):
    def __init__(self, params, data_mode):
        self.params = params
        self.mode = data_mode
        label_df = pd.read_csv(
            os.path.join(params['data']['root_dir'], 'labels_left.csv')
        )
        self.record_list = label_df.loc[label_df[f'split_{self.mode}'] == 1].to_dict('records')
        self.cxr_dir = os.path.join(params['data']['root_dir'])
        self.transform = self._get_transform(self.mode, self.params)

    def __getitem__(self, index):
        record_dict = self.record_list[index]
        cxr_path = os.path.join(self.cxr_dir, 'image', record_dict['folderid'] + '.jpeg')
        image_data = Image.open(cxr_path).convert('RGB')
        image_data = self.transform(image_data)

        if self.mode in ['train', 'valid', 'test']:
            gt_array = np.array([record_dict['dfa'], record_dict['pta'], record_dict['hka']])
            return {
                'image': image_data,
                'gt': gt_array,
                'cxr_file_name': record_dict['folderid']
            }
        else:
            # external without gt.
            raise NotImplementedError

    def __len__(self):
        return len(self.record_list)

    @staticmethod
    def _get_transform(mode, params):
        # use imagenet mean,std for normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        scale_size = params['data']['scale_size']
        crop_size = params['data']['crop_size']

        if mode == 'train':
            data_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif mode in ['valid', 'test', 'external']:
            data_transforms = transforms.Compose([
                transforms.Resize(scale_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            raise NotImplementedError

        return data_transforms
