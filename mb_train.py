import sys
sys.path.append('core')

import torch
import numpy as np
import argparse
from configs.motion import get_cfg
# from core.utils.misc import process_cfg 
# from core.FlowFormer import build_flowformer
# import datasets
from tqdm import tqdm
import pytorch_lightning as pl
from mb_model import MotionFormer
from mb_datasets import FlyingChairsDataModule
from configs.motion import get_cfg
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    )

torch.manual_seed(1234)
np.random.seed(1234)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-chk','--checkpoint',type=str, help='Path to model checkpoint') # defined in config file
    # parser.add_argument('--root', type=str, default= , help="Path to ")
    # args = parser.parse_args()

    cfg = get_cfg()
    # print(cfg)

    model = MotionFormer(cfg)
    dm = FlyingChairsDataModule('/share_chairilg/data/flyingchairs/', cfg)

    trainer = pl.Trainer(
            max_steps=cfg.mb_train.num_steps,
            accelerator="gpu",
            devices=1,
            val_check_interval=cfg.mb_train.val_inter,
            callbacks=[
                LearningRateMonitor("step"),
                EarlyStopping(
                    monitor=cfg.mb_train.monitor,
                    mode=cfg.mb_train.mode,
                    patience=cfg.mb_train.patience,
                ),
                ModelCheckpoint(
                    # dirpath=cfg.mb_train.chk_path
                    filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                    monitor=cfg.mb_train.monitor,
                    mode=cfg.mb_train.mode,
                )
            ],
            # prepare_data_per_node=False,
    )

    trainer.fit(
        model=model,
        datamodule=dm
    )

    # metric = trainer.callback_metrics["valid_mAP"]
    # print("Valid mAP", metric)

    # test run
    # val_dataset = datasets.FlyingChairs(split='val', root='/share_chairilg/data/flyingchairs/')
    # val_loader = torch.utils.
    # with torch.no_grad():
    #     model.eval()
    #     epe_list = []
    #     for val_id in tqdm(range(len(val_dataset))):
    #         image1, image2, flow_gt, _ = val_dataset[val_id]
    #         image1 = image1[None].cuda()
    #         image2 = image2[None].cuda()

    #         mb = model(image1, image2)

    #         # print(mb.shape)
    #         # epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
    #         # epe_list.append(epe.view(-1).numpy())
    #         break
    #     # Mean of L2 norm
    #     # epe = np.mean(np.concatenate(epe_list))
    #     epe=0
    #     print("Validation Chairs EPE: %f" % epe)

    # trainer = pl.Trainer()
    # preds = trainer.predict(model, val_dataset)