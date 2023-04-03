import sys
sys.path.append('core')

import torch
import numpy as np
import argparse
from configs.motion import get_cfg
from core.utils.misc import process_cfg 
from core.FlowFormer import build_flowformer
from unet_resnet import UNet
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class MotionFormer(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.cfg = get_cfg() # pull configurations
        print(self.cfg)

        self.backbone = torch.nn.DataParallel(build_flowformer(self.cfg))
        # self.backbone = build_flowformer(self.cfg)
        self.backbone.load_state_dict(torch.load(self.cfg.model))
        self.backbone.cuda()
        # input - (flow, img) 5D; output - motion or not (0, 1)
        self.head = UNet(5, 2) 
        self.head.cuda()

        self.loss = torch.nn.CrossEntropyLoss()
        self.map = MeanAveragePrecision() # eval metric


    def forward(self, img1, img2):
        flow_pre, _ = self.backbone(img1, img2)
        mb = self.head(torch.concat((flow_pre, img1), dim=1)) # concatenate along channel dim
        print(mb.shape) ##!!
        return mb


    def training_step(self, batch, batch_idx):
        img1, img2, motion_gt, _ = batch
        mb_pred = self(img1, img2)

        # loss = self.loss(mb_pred, motion_gt)
        # self.log('train_loss', loss)

        # self.map(mb_pred, motion_gt)
        # self.log('train_mAP', self.map)

        # return loss
        return 0
    
    def validation_step(self, batch, batch_idx):
        img1, img2, motion_gt = batch
        mb_pred = self(img1, img2)
        
        loss = self.loss(mb_pred, motion_gt)
        self.log('valid_loss', loss)

        self.map(mb_pred, motion_gt)
        self.log('valid_mAP', self.map)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                                     self.parameters(),
                                     lr=self.cfg.mb_train.lr,
                                    )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                optimizer,
                                                                mode='max',
                                                                factor=self.cfg.mb_train.sch_factor,
                                                                patience=self.cfg.mb_train.patience,
                                                                )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor":self.monitor, # only for scheduler like ReduceLROnPlateau
        }

        return {
                "optimizer":optimizer,
                "lr_scheduler":lr_scheduler_config,
            }


