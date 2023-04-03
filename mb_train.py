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
from unet_resnet import UNet

torch.manual_seed(1234)
np.random.seed(1234)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-chk','--checkpoint',type=str, help='Path to model checkpoint')
    args = parser.parse_args()

    cfg = get_cfg()
    # process_cfg(cfg) ##!! not sure if required
    print(cfg)

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))
    model.cuda()

    # input - (flow, img) 5D; output - motion or not (0, 1)
    head = UNet(5, 2) 
    head.cuda()

    # test run
    val_dataset = datasets.FlyingChairs(split='val', root='/share_chairilg/data/flyingchairs/')
    with torch.no_grad():
        model.eval()
        epe_list = []
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            flow_pre, _ = model(image1, image2)

            print(flow_pre.shape, image1.shape)
            # flow_pre = torch.permute(flow_pre.squeeze(), (1, 2, 0))
            # input = torch.concat((
            #                         flow_pre.squeeze(),
            #                         torch.permute(image1, (2, 0, 1)) 
            #                       ) ,dim=0)

            mb = head(
                        torch.concat((flow_pre, image1), dim=1)
                    )

            # epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
            # epe_list.append(epe.view(-1).numpy())
            break
        # Mean of L2 norm
        # epe = np.mean(np.concatenate(epe_list))
        epe=0
        print("Validation Chairs EPE: %f" % epe)

