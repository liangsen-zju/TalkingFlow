import torch
import torch.nn as nn
from pathlib import Path

from src.optims import get_optimizer, get_scheduler
from src.utils.landmark import landmarks_to_video_with_gt
from src.utils.debug import debug

from .base import Base
from .talkingFlow import TalkingFlow
        

class TalkingFlowModel(Base):

    def __init__(self, opt, logger):
        Base.__init__(self, opt, logger)

        self.opt = opt
        self.logger = logger
        self.names_model = ["G"]                           # G

        # define networks
        self.netG = TalkingFlow(opt)
        self.netG.to(f"cuda:{opt.GPUS[0]}")
        self.netG = torch.nn.DataParallel(self.netG, opt.GPUS)    # multi-GPUs


        # init
        if self.opt.MODEL.pretrain or self.opt.TRAIN.resume:
            self.netG.module.set_actnorm_init(True)
            # self.netG.set_actnorm_init(True)

        else:
            self.netG.module.set_actnorm_init(False)
            # self.netG.set_actnorm_init(False)

        if self.isTrain:
            self.names_loss = opt.LOSS.names

            # define optimizers
            self.optimizer_G = get_optimizer(self.netG, opt)
            self.optimizers.append(self.optimizer_G)
            self.scheduler_G = get_scheduler(self.optimizer_G, opt)
            self.schedulers.append(self.scheduler_G)

    def set_input(self, batch_input, contain_gt=True):
        
        if contain_gt:
            self.data = batch_input['data'].float().to(self.device)      # (nbatch,  nsteps=60, dim_motion=136)
            self.cond = batch_input["cond"].float().to(self.device)      # (nbatch, nsteps=173, dim_condit=80)
            
            self.data = self.data.permute(0, 2, 1)                       # (nbatch, 136, 60)
            self.cond = self.cond.permute(0, 2, 1)                       # (nbatch,  80, 173)

            self.B = self.data.size(0)
            debug(self.opt, f"data = {self.data.shape}, cond = {self.cond.shape}, dtype={self.data.dtype}", verbose=False)
        
        else:
            self.cond = batch_input["cond"].float().to(self.device)      # (nbatch, nsteps=173, dim_condit=80)
            self.cond = self.cond.permute(0, 2, 1)                       # (nbatch,  80, 173)

            self.B = self.cond.size(0)
            debug(self.opt, f"cond = {self.cond.shape}, dtype={self.cond.dtype}", verbose=False)
        

    
    def remove_weightnorm(self):
        def remove(conv_list):
            new_conv_list = nn.ModuleList()
            for old_conv in conv_list:
                old_conv = nn.utils.remove_weight_norm(old_conv)
                new_conv_list.append(old_conv)
            return new_conv_list

        for model in self.netG.module.FusionNets:
            model.layer_head = nn.utils.remove_weight_norm(model.layer_head)
            model.layer_cond = nn.utils.remove_weight_norm(model.layer_cond)
            model.in_layers = remove(model.in_layers)
            model.res_skip_layers = remove(model.res_skip_layers)


    def training(self, batch_input):
        self.train()     # set train model
        self.set_input(batch_input) 

        # forward
        z, logdet = self.netG(self.data, self.cond)
        debug(self.opt, f"[Train] netG z shape = {len(z)}, logs shape = {logdet.shape}", verbose=False)

        # loss
        # loss, self.loss_dict = self.netG.module.eval_losses()
        loss, self.loss_dict =  self.netG.module.loss_function_(z, logdet)
        debug(self.opt, f"[Train] loss shape = {loss.shape}", verbose=False)
        
        # backward
        self.optimizer_G.zero_grad()
        loss.backward()

        if self.opt.TRAIN.grad_clip_norm:  # operate grad
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.opt.TRAIN.max_grad_clip)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt.TRAIN.max_grad_norm)

        # step
        self.optimizer_G.step()

        return loss


    def validation(self, batch_input):
        self.eval()      # set eval model
        self.set_input(batch_input)

        with torch.no_grad():
            z, logdet = self.netG(self.data, self.cond)
            loss, self.loss_dict =  self.netG.module.loss_function_(z, logdet)
        return loss
        

    def testing(self, batch_input, cur_epoch=0, cur_batch=0, contain_gt=True, save_animation=True):
        self.eval()                 # set eval model
        self.set_input(batch_input, contain_gt)

        with torch.no_grad():
            pred_x, _ = self.netG.module.inverse(self.cond)             # B x dim_motion x nsteps
            pred_x = pred_x.permute(0, 2, 1)                            # B x nsteps x dim_motion
            

            if save_animation:
                # save generated data
                nsteps = min(pred_x.size(1), self.data.size(2))
                motion_gt = self.data.permute(0,2,1)[:, :nsteps, :136]
                motion_pt = pred_x[:, :nsteps, :136]
                landmarks_to_video_with_gt(motion_pt, motion_gt, 
                                    path_save=Path(self.opt.OUTPUT_DIR), 
                                    suffix=f'sampled_{cur_epoch:05d}_{cur_batch:03d}', 
                                    w=256, h=256, dpi=100, maxsave=3)

        return pred_x


    def testing2(self, batch_input, cur_epoch=0, cur_batch=0):
        self.eval()    # set eval model
        self.set_input(batch_input)


        with torch.no_grad():

            pred_xs = []
            nsteps = self.cond.size(2)     # 3x136x60, 3x80x173
            step = 173
            rato = 173 / 60

            for i in range(0, nsteps-step, 3):
                
                icond = self.cond[:, :, i:i+step]
                pred_x, _ = self.netG.module.inverse(icond)                   # B x dim_motion x step
                pred_x = pred_x.permute(0, 2, 1)                              # B x step x dim_motion
                debug(self.opt, f"[testing] i={i}, pred_x={pred_x.shape}, icond = {icond.shape}", verbose=True)

                if i == 0:
                    pred_xs.append(pred_x)
                else:
                    pred_xs.append(pred_x[:, -1:, :])


            pred_xs = torch.cat(pred_xs, dim=1)
            debug(self.opt, f"[testing] pred_xs={pred_xs.shape}", verbose=True)

            # save generated data
            nsteps = min(pred_xs.size(1), self.data.size(2))

            motion_pt = pred_xs[:, :nsteps, :136]
            motion_gt = self.data.permute(0,2,1)[:, :nsteps, :136]
            landmarks_to_video_with_gt(motion_pt, motion_gt, 
                                path_save=Path(self.opt.OUTPUT_DIR), 
                                suffix=f'sampled_{cur_epoch:05d}_{cur_batch:03d}', 
                                w=256, h=256, dpi=100, maxsave=3)