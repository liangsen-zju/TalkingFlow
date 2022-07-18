import pprint
import argparse
import numpy as np
from pathlib import Path

import torch
torch.autograd.set_detect_anomaly(True) 

from src.models import create_model, calculate_model_params
from src.config import config as opt
from src.config import update_config
from src.utils.landmark import landmarks_to_video
from src.utils.logger import Logger



def parese_args():
    pareser = argparse.ArgumentParser(description='TalkingFlow Network')

    # general 
    pareser.add_argument("--opt", type=str, default="experiments/xxx.yaml", help="experiment configure file name")
    args, _ = pareser.parse_known_args()
    update_config(args.opt)

    pareser.add_argument("--gpus", type=str, default="0", help="gpus")
    args = pareser.parse_args()

    return args
    

if __name__ == "__main__":
    args = parese_args()
    path_config = Path(args.opt)
    postfix = opt.TRAIN.resume_suffix if opt.TRAIN.resume else None    # if resume, resume to resume_suffix folder
    logger = Logger(opt.OUTPUT_DIR, path_config=path_config, postfix=postfix) 
    
    logger.info(pprint.pformat(opt))
    opt.OUTPUT_DIR = logger.path_output


    path_data = Path("/mfs/lsen/DATA/Obama/processing")

    vid = "Celebrating-Independence-Day"
    path_audio  = path_data.joinpath(vid, "audio/audio_trim.wav")
    path_speech = path_data.joinpath(vid, "audio/feat_mel_no_align_trim.npy")
    path_motion = path_data.joinpath(vid, "audio/feat_ldmk_trim_norm.npy")

    data_speech = np.load(path_speech)      # (11295, 80)
    # data_motion = np.load(path_motion).reshape(-1, 136)      # (3930, 136)
    print(f"data_speech={data_speech.shape}")

    # nscends = 20  
    # fps = 30
    # data_speech = data_speech[: int(nscends*fps*len(data_speech)/len(data_motion))+1, ...]  # (863, 80)
    # data_motion = data_motion[: nscends*fps, ...]                                           # (300, 136)
    # print(f"trim data_speech={data_speech.shape}, data_motion={data_motion.shape}")

    # add to batch_input
    batch_input = {}
    # batch_input['data'] = torch.from_numpy(data_motion).unsqueeze_(0)
    batch_input['cond'] = torch.from_numpy(data_speech).unsqueeze_(0)

    # logger.info(f"Test dataset size = {len(test_dataset)}, batch = {len(test_loader)}")
    device = torch.device(f'cuda:{opt.GPUS[0]}')  # get device name: CPU or GPU

    # get model
    model = create_model(opt, logger) 
    model.setup(opt)                    # load model
    # model.remove_weightnorm()


    # Add for WaveFlow
    # model.netG.module = model.netG.module.remove_weightnorm(model.netG.module)    
    # for i, batch_input in enumerate(test_loader):

    # motion = batch_input['data'].float().to(device)           # (nbatch,  nsteps=60, dim_motion=136)
    speech = batch_input['cond'].float().to(device)             # (nbatch, nsteps=173, dim_condit=80)
    # motion = motion.permute(0, 2, 1)                          # (nbatch, 136, 60)
    speech = speech.permute(0, 2, 1)                            # (nbatch,  80, 173)
    # B = motion.size(0)
    # motion_gt = motion.permute(0,2,1)[0:1, :, :]         


    data_pt = model.testing(batch_input, cur_epoch=opt.MODEL.load_epoch, save_animation=False, contain_gt=False)[0]  # B x nsteps x 136
    data_pt = data_pt.cpu().numpy()

    # save 
    print("data_pt shape = ", data_pt.shape)    # nsteps x 136
    np.save(f"data/ldmk_pdt_align30.npy", data_pt)

    # save 
    fps=29.97003
    data_pt = data_pt[ :int(15*fps+0.5)]

    path_video=Path(opt.OUTPUT_DIR).joinpath(f'sampled_{opt.MODEL.load_epoch:05d}.mp4')
    landmarks_to_video(data_pt, path_video, path_audio, fps=fps, w=256, h=256, dpi=80)
    print(f"SVAE -> {opt.OUTPUT_DIR}")


