import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from os import path, cpu_count
from PIL import Image
import numpy as np
from torchvision.transforms import ColorJitter, Compose
import cv2
import pytorch_lightning as pl

from configs.motion import get_cfg ##!! only for exp - remove later

class FlyingChairs(Dataset):
    """
    Data reader for image pair and motion boundaries.
    """

    def __init__(self, root, split, transform=None):
        # self.root = root
        # self.split = split
        self.transform = transform
        self.images_0 = sorted(glob(path.join(root, split, '*img_0.png')))
        self.images_1 = sorted(glob(path.join(root, split, '*img_1.png')))
        self.motions = sorted(glob(path.join(root, split, '*mb_10.png')))
        self.flows = sorted(glob(path.join(root, split, '*10.flo')))

        assert len(self.images_0) == len(self.motions), "Images must be twice as much as motion boundary predictions"

    def __len__(self):
        ''' Specific to output '''
        return len(self.motions)
        
    def __getitem__(self, idx):
        img0 = np.array(Image.open(self.images_0[idx])).astype(np.uint8)
        img1 = np.array(Image.open(self.images_1[idx])).astype(np.uint8)
        mb = np.array(Image.open(self.motions[idx])).astype(np.float32)/255

        if self.transform:
            img0, img1, mb = self.transform(img0, img1, mb)

        # conver to tensors
        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        mb = torch.from_numpy(mb).unsqueeze(0).float()

        return img0, img1, mb


class MotionAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, pwc_aug=False, noise=False):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        self.pwc_aug = pwc_aug
        if self.pwc_aug:
            print("[Using pwc-style spatial augmentation]")

        # noise aug params
        self.noise = noise

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, motion):
        ''' Random resize and cropping '''

        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            motion = cv2.resize(motion, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            # motion = motion * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                motion = motion[:, ::-1] #* [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                motion = motion[::-1, :] #* [1.0, -1.0]

        if img1.shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        if img1.shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        motion = motion[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, motion

    def __call__(self, img1, img2, motion):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        if self.pwc_aug:
            raise NotImplementedError
        else:
            img1, img2, motion = self.spatial_transform(img1, img2, motion)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        motion = np.ascontiguousarray(motion)
        
        if self.noise:
            stdv = np.random.uniform(0.0, 5.0)
            img1 = (img1 + stdv * np.random.randn(*img1.shape)).clip(0.0, 255.0)
            img2 = (img2 + stdv * np.random.randn(*img2.shape)).clip(0.0, 255.0)

        return img1, img2, motion



class FlyingChairsDataModule(pl.LightningDataModule):

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.aug_params = {'noise':cfg.add_noise, 'crop_size': cfg.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        self.train_transform = MotionAugmentor(**self.aug_params) if cfg.augment else None
        self.save_hyperparameters()

    def setup(self, stage:str):
        if stage == 'fit':
            self.train_set = FlyingChairs(self.root, 'train', self.train_transform)
            self.val_set = FlyingChairs(self.root, 'val') # no additional augmentation
        elif stage == 'predict':
            self.val_set = FlyingChairs(self.root, 'val')
        ##!! test on sintel?

    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          self.cfg.batch_size, #| self.hparams.batch_size,
                          shuffle=True,
                          num_workers=min(cpu_count(), self.cfg.batch_size))

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                          self.cfg.batch_size, #| self.hparams.batch_size,
                          shuffle=True,
                          num_workers=min(cpu_count(), self.cfg.batch_size))

    

if __name__ == '__main__':
    cfg = get_cfg()
    # aug_params = {'noise':cfg.add_noise, 'crop_size': cfg.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
    # train_transform = MotionAugmentor(**aug_params)

    # # give tensor images by default - no need for additional augmentation
    # train_data = FlyingChairs('/share_chairilg/data/flyingchairs/', 'val', train_transform)

    dm = FlyingChairsDataModule('/share_chairilg/data/flyingchairs/', cfg)
    dm.setup('fit')