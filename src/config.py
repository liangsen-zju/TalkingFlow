import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.LOG_PREFIX = ''
config.GPUS = '1'
config.WORKERS = 8
config.DEBUG = False

# pxxxx
config.isTrain = True
config.preprocess = ''
config.verbose = True

# frequence
config.FREQ = edict()
config.FREQ.batch_save = 20
config.FREQ.batch_print = 1
config.FREQ.batch_display = 100
config.FREQ.batch_test  = 100000

config.FREQ.epoch_save = 1
config.FREQ.epoch_print = 1
config.FREQ.epoch_display = 10
config.FREQ.epoch_valid = 1
config.FREQ.epoch_test  = 1

config.FREQ.update_html = 1


# visual
config.VISUAL = edict()
config.VISUAL.use_HTML = True
config.VISUAL.use_visdom = True

config.VISUAL.vsdm_id = 0
config.VISUAL.vsdm_wsize = 256
config.VISUAL.vsdm_port = 8098
config.VISUAL.vsdm_ncol = 3
config.VISUAL.vsdm_server = "http://localhost"
config.VISUAL.vsdm_env = "main-2"

# cudnn related params
config.CUDNN = edict()
config.CUDNN.benchmark = True
config.CUDNN.deterministic = True
config.CUDNN.enable = True


##########################
# model related params
config.MODEL = edict()
config.MODEL.name = "LandmarkAE"
config.MODEL.dropout = 0.1

# load & init model
config.MODEL.pretrain = False
config.MODEL.path_pretrained = ""
config.MODEL.path_checkpoint = ""
config.MODEL.load_epoch = 0

# for MOGLOW
config.MODEL.dim_motion = 68*2
config.MODEL.dim_speech = 27
config.MODEL.dim_headpose = 6
config.MODEL.dim_hidden = 256
config.MODEL.n_flowstep = 12
config.MODEL.n_lookprevs = 7
config.MODEL.n_lookahead = 7
config.MODEL.n_layer_lstm = 2
config.MODEL.n_layer_fpn = 4
config.MODEL.n_layer_tcn = 4
config.MODEL.n_layer_flow = 12

config.MODEL.n_flows_per_fpn = [3, 3, 3, 3]

config.MODEL.lambda_std = [1, 1, 1, ]

# for actnet
config.MODEL.dim_input  = 68*2
config.MODEL.dim_output = 68*2
config.MODEL.dim_latent = 64
config.MODEL.dim_hidden = 128

# for transformer
config.MODEL.n_src_vocab = 0
config.MODEL.n_tgt_vocab = 0
config.MODEL.src_pad_idx = 0
config.MODEL.tgt_pad_idx = 0

# for wavenet
config.MODEL.n_early_every = 0
config.MODEL.n_early_size = 0
config.MODEL.n_layer_wave = 0
config.MODEL.kernel_size = 0


config.MODEL.d_input = 68*2
config.MODEL.d_output = 68*2
config.MODEL.d_embed = 512 
config.MODEL.d_model = 512
config.MODEL.d_latent = 256
config.MODEL.d_inner = 2048
config.MODEL.d_k = 64
config.MODEL.d_v = 64
config.MODEL.n_head = 8
config.MODEL.n_layers = 6
config.MODEL.max_position = 200

# In section 3.4 of paper "Attention Is All You Need", there is such detail:
# "In our model, we share the same weight matrix between the two
# embedding layers and the pre-softmax linear transformation...
# In the embedding layers, we multiply those weights by \sqrt{d_model}".
# Options here:
#   'emb': multiply \sqrt{d_model} to embedding output
#   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
#   'none': no multiplication
config.MODEL.scale_emb_or_prj = 'prj'   
config.MODEL.scale_emb = False
config.MODEL.trg_emb_prj_weight_sharing = True
config.MODEL.emb_src_trg_weight_sharing = True


##########################
# dataset related params
config.DATASET = edict()
config.DATASET.name = 'ravdess'
config.DATASET.type = ''             # "clips" | "frame" | "none"
config.DATASET.root = "../data/"
config.DATASET.manifest_train = "data/train_manifest.csv"
config.DATASET.manifest_valid = "data/valid_manifest.csv"
config.DATASET.manifest_test = "data/test_manifest.csv"
config.DATASET.manifest_infer = "data/infer_manifest.csv"
config.DATASET.num_train_imgs = 16

# from dataset iteratiion
config.DATASET.nwind_second_train = 0
config.DATASET.nwind_second_valid = 0
config.DATASET.nwind_second_test = 0

# augmentation
config.DATASET.flip = False

#############################
# Trainning related parameter
config.TRAIN = edict()

config.TRAIN.apply_validation = False
config.TRAIN.apply_testing = False
config.TRAIN.apply_update_parameters = False

config.TRAIN.lr = 0.001
config.TRAIN.lr_step = [50, 100]
config.TRAIN.lr_factor = 0.1
config.TRAIN.beta1 = 0.5
config.TRAIN.beta2 = 0.999
config.TRAIN.lambda1 = 100
config.TRAIN.lr_policy = ''
config.TRAIN.lr_decay_iters = 0
config.TRAIN.lr_milestones = [10,100,1000]
config.TRAIN.grad_clip_norm = False
config.TRAIN.max_grad_clip = 0
config.TRAIN.max_grad_norm = 0

# for motionVAE
config.TRAIN.p_use_pt = 0.5

config.TRAIN.reset_lr = False
config.TRAIN.resume = False
config.TRAIN.resume_suffix = ""
config.TRAIN.checkpoints = ""
config.TRAIN.batch_size = 24
config.TRAIN.shuffle = True
config.TRAIN.epoch_begin = 0
config.TRAIN.epoch_end = 100
config.TRAIN.niter = 0
config.TRAIN.niter_decay = 100
config.TRAIN.load_iter = 0
config.TRAIN.epoch = 0

# for cyclical schedule
config.TRAIN.n_cycle = 5
config.TRAIN.ratio = 0.8
config.TRAIN.beta_start= 0.0
config.TRAIN.beta_stop = 1.0

config.TRAIN.optimizer = 'adam'
config.TRAIN.momentum = 0.9
config.TRAIN.weight_decay = 0.0001
config.TRAIN.nesterov = False
config.TRAIN.gamma1 = 0.99
config.TRAIN.gamma2 = 0.0

###############################
# loss related params
config.LOSS = edict()
config.LOSS.names = ["all", "L1"]
config.LOSS.lambda_rec = 1
config.LOSS.lambda_kld = 0
config.LOSS.lambda_dif = 0
config.LOSS.lambda_cyc = 0
config.LOSS.lambda_trp = 0
config.LOSS.lambda_geo = 0
config.LOSS.lambda_rec_smooth = 0
config.LOSS.lambda_geo_smooth = 0
config.LOSS.lambda_logp = 0
config.LOSS.lambda_logdet = 0

#############################
# TEST related params
config.TEST = edict()
config.TEST.batch_size = 24
config.TEST.path_input = ""
config.TEST.path_save = ""

#############################
# INFER related params
config.INFER = edict()
config.INFER.batch_size = 24
config.INFER.path_input = ""
config.INFER.path_save = ""

#############################
# DEBUG related params
config.DEBUG = edict()
config.DEBUG.debug = False
config.DEBUG.save_batch_image_gt = False
config.DEBUG.save_batch_image_pred = False
config.DEBUG.save_heatmaps_gt = False
config.DEBUG.save_heatmaps_pred = False


def update_index_exp(index_exp):
    # prepare index
    str_ids = index_exp.split(',')

    result = []
    for i in range(int(len(str_ids) / 2)):
        start = int(str_ids[ 2*i])
        end = int(str_ids[2*i + 1])

        if end > start:
            result += range(start, end)

    # print('indexes', result)

    return result


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x for x in v['STD']])

        # if "expression_indx" in v and v['expression_indx']:
        #     v['expression_indx'] = update_index_exp(v['expression_indx'])

    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(v['EXTRA']['HEATMAP_SIZE'])
        
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])

    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))

        for k, v in exp_config.items():

            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
    
    # update some config item
    config.GPUS = [int(i) for i in config.GPUS.split(',')]


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)



if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
