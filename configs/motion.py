from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'default'
_CN.suffix ='arxiv2'
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 2
_CN.sum_freq = 100
_CN.val_freq = 499999999
_CN.image_size = [384, 512] #[368, 496]
_CN.add_noise = True
_CN.critical_params = []
_CN.augment = True

_CN.transformer = 'latentcostformer'
_CN.model = 'checkpoints/things_kitti.pth'
# _CN.restore_ckpt = 'checkpoints/sintel.pth'

###########################################
# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.predictor_dim = 128
_CN.latentcostformer.motion_feature_dim = 209 # use concat, so double query_latent_dim
_CN.latentcostformer.arc_type = 'transformer'
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.vertical_encoder_attn = "twins"
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.gma = "GMA"
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.pwc_aug = False
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.no_sc = False
_CN.latentcostformer.use_rpe = False
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False
# decoder
_CN.latentcostformer.decoder_depth = 12
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']
##########################################

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 50000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'

### MOTIONB TRAINER
_CN.mb_train = CN()
_CN.mb_train.lr = 3e-4
# _CN.mb_train.adamw_decay = 1e-5
# _CN.mb_train.epsilon = 1e-8
_CN.mb_train.num_steps = 50000
_CN.mb_train.sch_factor = 0.9
_CN.mb_train.monitor = 'valid_map'
_CN.mb_train.mode = 'max'
_CN.mb_train.patience = 5000 # always relative to val_inter
_CN.mb_train.val_inter = 500
_CN.mb_train.chk_path = 'lightning_logs'

def get_cfg():
    return _CN.clone()
