# TODO: program for list creation and dataset like I like

from albumentations.pytorch import ToTensor
import configparser
import os
import os.path as osp
from scipy.misc import imread
import torch
import rorch.nn as nn
from torch.utils.data import Dataset


def process_config(seqfile):

    config = configparser.ConfigParser()
    config.read(seqfile)
    
    seq_dir = config['Sequence']['imDir']
    seq_ext = config['Sequence']['imExt']

    c_width = int(config['Sequence']['imWidth'])
    c_height = int(config['Sequence']['imHeight'])
    im_shape = (c_height, c_width, 3)

    cam_motion = bool(config['Sequence'].get('cam_motion', False))

    return seq_dir, seq_ext, im_shape, cam_motion
    

class SequencesPseudoDataset(Dataset):

    def __init__(self, base_dir, dset='train', start_ind=0):
        mot_dir = osp.join(base_dir, dset)
        mot_seq = os.listdir(mot_dir)[start_ind:]
        self.mot_paths = sorted([osp.join(mot_dir, seq) for seq in mot_seq])

    def __len__(self):
        return len(self.mot_paths)

    def __getitem__(self, ind):
        mot_p = self.mot_paths[ind]
        seqfile = osp.join(mot_p, 'seqinfo.ini')

        seq_dir, seq_ext, im_shape, cam_motion = process_config(seqfile)

        im_path = osp.join(mot_p, seq_dir)
        seq_images = sorted(glob(osp.join(im_path, '*' + seq_ext)))

        return seq_images, im_shape, cam_motion, im_path


def normalize(image, mean, std):
    return (image - mean[:, None, None]) / std[:, None, None]

def build_image_transform(min_size, max_size, image_mean, image_std):
    if min_size is not None:
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
    
    def image_transform(image, target=None):

        if len(image.shape) != 3:
            raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")

        image = normalize(image, image_mean, image_std)
        #if min_size is not None and max_size is not None:
        #    image, target = resize(image, target, min_size, max_size)

        return image, target


class SequenceImagesPseudoDataset(Dataset):

    def __init__(self, seq_images, transform=None):

        self.seq_images = seq_images
        self.transform = transform

    def __getitem__(self, ind):
        im = imread(self.seq_images[ind])

        transf = ToTensor()
        image = transf(image=im)['image']

        if self.transform is not None:
            image, _ = self.transform(image)

        return torch.unsqueeze(image, 0)
    
    def __len__(self):
        return len(self.seq_images)


