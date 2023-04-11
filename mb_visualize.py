import sys
sys.path.append('core')

import torch
import numpy as np
import argparse
from configs.motion import get_cfg
from tqdm import tqdm
import pytorch_lightning as pl
from mb_model import MotionFormer
from mb_datasets import FlyingChairsDataModule, FlyingChairs
from configs.motion import get_cfg
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

if __name__ == '__main__':
    cfg = get_cfg()
    model = MotionFormer.load_from_checkpoint('/home/nikhil/Desktop/FlowFormer-Official/lightning_logs/epoch=3-valid_loss=0.14-valid_map=0.30.ckpt', 
                                              cfg=cfg)
    model.eval()
    # dm = FlyingChairsDataModule('/share_chairilg/data/flyingchairs/', cfg)
    dataset = FlyingChairs('/share_chairilg/data/flyingchairs/', 'val')
    disp_num=4; col=3


    fig = plt.figure(figsize=(30, 35))

    for i in range(disp_num):
        idx = np.random.randint(0, len(dataset))
        img1, img2, motion_gt = dataset[idx]

        with torch.no_grad():
            motion_pred = model(img1.unsqueeze(0), img2.unsqueeze(0)).squeeze(0)
            soft_pred = torch.softmax(motion_pred, dim=0)
            
            
        pred = soft_pred.cpu().numpy()
        mb = pred[1,...].reshape(pred.shape[1:])
        # mb = np.argmax(pred, axis=0).reshape(pred.shape[1:])

        # print(pred.shape, np.unique(pred))

        # hard_mb = np.argmax(motion_pred, axis=0).reshape(motion_pred.shape[1:])
        # print(hard_mb.shape, np.unique(hard_mb))
        fig.add_subplot(disp_num, col, col*i+1)
        plt.imshow(img1.permute((1, 2, 0)).numpy().astype(np.uint8))
        plt.title(f"Image Idx {idx}")
        plt.axis("off")

        fig.add_subplot(disp_num, col, col*i+2)
        plt.imshow(motion_gt)
        plt.title("Motion GT")
        plt.axis("off")    

        fig.add_subplot(disp_num, col, col*i+3)
        plt.imshow(mb*255)
        plt.title("Motion Pred")
        plt.axis("off")    

    
    plt.savefig('viz_results/mf_fchairs.png', bbox_inches='tight')
    plt.plot()

