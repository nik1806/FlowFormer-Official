import sys
sys.path.append('core')

import torch
import numpy as np
import argparse
from configs.motion import get_cfg
from core.utils.misc import process_cfg 
from core.FlowFormer import build_flowformer
import datasets
from tqdm import tqdm
from mb_model import MotionFormer
import pytorch_lightning as pl

torch.manual_seed(1234)
np.random.seed(1234)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-chk','--checkpoint',type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    model = MotionFormer()    

    # test run
    val_dataset = datasets.FlyingChairs(split='val', root='/share_chairilg/data/flyingchairs/')
    # val_loader = torch.utils.
    
    with torch.no_grad():
        model.eval()
        epe_list = []
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            mb = model(image1, image2)

            # print(mb.shape)
            # epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
            # epe_list.append(epe.view(-1).numpy())
            break
        # Mean of L2 norm
        # epe = np.mean(np.concatenate(epe_list))
        epe=0
        print("Validation Chairs EPE: %f" % epe)

    # trainer = pl.Trainer()
    # preds = trainer.predict(model, val_dataset)