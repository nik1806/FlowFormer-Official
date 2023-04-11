import sys
sys.path.append('core')

import torch
import numpy as np
import argparse
from core.utils.misc import process_cfg 
from core.FlowFormer import build_flowformer
from unet_resnet import UNet
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import RetrievalMAP
from torchmetrics.classification import AveragePrecision
from configs.motion import get_cfg
from collections import OrderedDict

class MotionFormer(pl.LightningModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg # pull configurations

        self.backbone = build_flowformer(self.cfg)
        loaded_dict = torch.load(self.cfg.model)

        temp_dict = OrderedDict()
        for key, item in loaded_dict.items():
            temp_dict.update({key.replace("module.",""): item})
        self.backbone.load_state_dict(temp_dict) # no training it

        # FlowFormer is frozen
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        
        # self.backbone.cuda()
        # input - (flow, img) 5D; output - motion or not (0, 1)
        # change input to just flow (2D - (x,y))
        self.head = UNet(2, 2) 
        # self.head.cuda()

        self.loss = torch.nn.CrossEntropyLoss()
        self.map = AveragePrecision(task='binary') #MeanAveragePrecision() # eval metric

        # self.map_list = []
        # self.valid_loss = []
        # self.valid_mAP = []

    def forward(self, img1, img2):
        flow_pre = self.backbone(img1, img2)
        # mb = self.head(torch.concat((flow_pre[0], img1), dim=1)) # concatenate along channel dim
        mb = self.head(flow_pre[0]) ##!! exp - only flow
        return mb

    def training_step(self, batch, batch_idx):
        img1, img2, motion_gt = batch
        mb_pred = self(img1, img2)

        loss = self.loss(mb_pred, motion_gt)
        self.log('train_loss', loss.cpu(), prog_bar=True)

        # map = self.map(mb_pred, motion_gt)
        # self.log('train_mAP', map)

        return {"loss":loss}
        # return 0
    
    def validation_step(self, batch, batch_idx):
        img1, img2, motion_gt = batch
        mb_pred = self(img1, img2)
        
        # print(mb_pred.shape, motion_gt.shape)

        loss = self.loss(mb_pred, motion_gt)
        # self.valid_loss.append(loss.cpu())
        self.log('valid_loss', loss.cpu(), prog_bar=True)

        self.map(torch.softmax(mb_pred, dim=1)[:,1,...], motion_gt) ##!! want prob or logit in single channel
        self.log('valid_map', self.map, prog_bar=True)

        # self.map.update()
        # self.map_list.append(map)

        return {"valid_loss":loss, "valid_map":self.map}
    
    def validation_epoch_end(self, outputs) -> None:
        # self.log("valid_loss_epoch", np.mean(self.valid_loss))
        # self.log("valid_map_epoch", np.mean(self.map_list))
        self.log("valid_loss_epoch", np.array([x["valid_loss"].cpu() for x in outputs]).mean(), prog_bar=True)
        self.log("valid_map_epoch", 
                 np.array([x["valid_map"].cpu() for x in outputs]).mean(), 
                 metric_attribute='valid_map',
                 prog_bar=True,
                 )


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                                     self.parameters(),
                                     lr=self.cfg.mb_train.lr,
                                    )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                optimizer,
                                                                mode=self.cfg.mb_train.mode,
                                                                factor=self.cfg.mb_train.sch_factor,
                                                                patience=self.cfg.mb_train.patience,
                                                                )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": self.cfg.mb_train.val_inter,
            "monitor":self.cfg.mb_train.monitor, # only for scheduler like ReduceLROnPlateau
        }

        return {
                "optimizer":optimizer,
                "lr_scheduler":lr_scheduler_config,
            }


if __name__ == '__main__':

    cfg = get_cfg() 
    # print(cfg)

    # backbone = torch.nn.DataParallel(build_flowformer(cfg))
    # backbone = build_flowformer(cfg)
    # backbone.load_state_dict(torch.load(cfg.model))

    model = build_flowformer(cfg)
    # model.load_state_dict(backbone.state_dict(), strict=True)
    print("loading complete")

    loaded_dict = torch.load(cfg.model)
    state_dict = OrderedDict()
    for key, item in loaded_dict.items():
        state_dict.update({key.replace("module.",""): item})

    model.load_state_dict(state_dict)
