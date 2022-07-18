import torch
import torch.nn as nn
import numpy as np

from src.utils.debug import debug

from .layers import ActNorm1d, Invertible1x1Conv1d as Invertible1x1Conv


# @torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class TalkingFlow(nn.Module):

    def __init__(self, opt):
        super(TalkingFlow, self).__init__()

        self.opt = opt
        self.nc_motion = opt.MODEL.dim_motion                   # 8 --> 136
        self.nc_condit = opt.MODEL.dim_speech                   # 80 

        self.n_flows = opt.MODEL.n_flowstep                     # 12
        self.n_early_every = opt.MODEL.n_early_every            # 4 --> 3

        self.lambda_std = np.array(opt.MODEL.lambda_std)                   # [1, 0.75, 0.5, 0.25]
        self.lambda_std = torch.from_numpy(self.lambda_std).float().cuda(opt.GPUS[0])

        kernel = 128
        nsteps_motion = 30 * self.opt.DATASET.nwind_second_train
        nsteps_speech = int(2.8740 * nsteps_motion + 0.5)           # 2.8740 = nsteps_segmt / nsteps_frame
        stride = (nsteps_motion * self.nc_motion - kernel) // (nsteps_speech -1) + 1
        self.upsample = nn.ConvTranspose1d(self.nc_condit, self.nc_condit, kernel_size=kernel, stride=stride)  
        debug(self.opt, f"\n[Model] nsteps_speech={nsteps_speech}, nsteps_motion={nsteps_motion}, stride={stride}", verbose=True)

        self.Conv1x1Invs = nn.ModuleList()
        self.FusionNets  = nn.ModuleList()
        self.Actnorms    = nn.ModuleList()
        nc_motion_rest = self.nc_motion                                       # 8 --> 136
        for k in range(self.n_flows):        
            if k % self.n_early_every == 0 and k > 0:                         # k =  3,  6,  9
                nc_motion_rest = nc_motion_rest - nc_motion_rest // 2         #   = 84, 42, 21

            self.Actnorms.append(ActNorm1d(num_channels=nc_motion_rest))        
            self.Conv1x1Invs.append(Invertible1x1Conv(nc_motion_rest))
            self.FusionNets.append(FusionNet(self.opt, nc_motion=nc_motion_rest - nc_motion_rest//2, 
                                                       nc_condit=self.nc_condit * self.nc_motion,
                                                       nc_output=2*(nc_motion_rest//2)))

            debug(self.opt, f"[Model] k={k} nc_motion_rest={nc_motion_rest}")

        self.nc_motion_rest = nc_motion_rest     # Useful during inference, 72 / 8


    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited
                debug(self.opt, f"[Model] set module {m.__class__.__name__} inited = {inited}")


    def upsampling_condition(self, c, nc_trim=None):
        """upsampling conditon 
        params:
            c shape = (B, dim_condit, nsteps_c), mel_spectorgram                                 (B, 80, 173)
        return:
            feat_c, (B, nc_motion*nc_speech, nsteps_x)
        """
        # preprocessing condition c
        feat_c = self.upsample(c)                                                                # (B, 80, 173) --> (B, 80, 8212)
        debug(self.opt, f"\t[Upsampling z] upsamle feat_c={feat_c.shape}", verbose=False)

        if nc_trim is not None:
            feat_c = feat_c[:, :, :nc_trim]                                                      # (B, 80, 8160=60*136)
        
        feat_c = feat_c.unfold(dimension=2, size=self.nc_motion, step=self.nc_motion)             # (B, 80, nsteps=60, nc_motion=136)
        debug(self.opt, f"\t[Upsampling z] unfold feat_c={feat_c.shape}", verbose=False)
        
        feat_c = feat_c.permute(0, 2, 1, 3)                                                       # (B, 60, 80, 136)
        feat_c = feat_c.contiguous().view(feat_c.size(0), feat_c.size(1), -1).permute(0, 2, 1)    # (B, 10880=80*136, 60)
        debug(self.opt, f"\t[Upsampling z] permute view feat_c={feat_c.shape}")
        
        return feat_c


    def forward(self, x, c):
        """For training, x --> z
            x shape = (B, dim_motion, nsteps_x), landmarks,         (B, 136, 60)
            c shape = (B, dim_condit, nsteps_c), mel_spectorgram    (B, 80, 173)
        """
        debug(self.opt, f"\n[Forward] start x shape={x.shape}, c={c.shape}", verbose=False)

        predt_z = []
        logdet = torch.zeros(c.size(0), dtype=c.dtype, device=c.device)
        feat_c = self.upsampling_condition(c, nc_trim=x.size(1) * x.size(2))                         # B x nc_condit x nsteps

        for k in range(self.n_flows):

            x, logdet = self.Actnorms[k](x, logdet)
            debug(self.opt, f"[Forward] k={k}, actnorm x shape = {x.shape} logdet={logdet.shape}", verbose=False)
  
            x, logdet = self.Conv1x1Invs[k](x, logdet)
            debug(self.opt, f"[Forward] k={k}, Conv1x1Invs x={x.shape}, logdet={logdet.shape}")
    
            # coupling layers
            nc_motion_half = x.size(1) - x.size(1) // 2
            x1, x2 = x[:, :nc_motion_half, :], x[:, nc_motion_half:, :]           
            debug(self.opt, f"[Forward] k={k}, coupling split x1={x1.shape}, x2={x2.shape}")
        
            feat = self.FusionNets[k](x1, feat_c)                        # (B, nc_motion_half, dim_output)
            shift, scale = feat[:, 0::2, :],  feat[:, 1::2, :]
            scale = torch.sigmoid(scale + 2.) + 1e-6                     # to make scale > 1, so that log(s) > 0
            x2 = (x2 + shift) * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2]) + logdet

            # multi-scale split 
            if k % self.n_early_every == (self.n_early_every -1) and k > 0 and k < self.n_flows -1:
                predt_z.append(x2)
                x = x1
                debug(self.opt, f"[Forward] k={k}, coupling muli-scale x={x.shape}, x2={x2.shape}, logdet={logdet.shape}", verbose=False)
            else:
                x = torch.cat([x1, x2], dim=1)
                debug(self.opt, f"[Forward] k={k}, coupling concat x={x.shape}, logdet={logdet.shape}", verbose=False)


        # append the last x
        predt_z.append(x) 
        predt_z = list(reversed(predt_z))
        # predt_z = torch.cat(predt_z, dim=1)

        return predt_z, logdet

    def inverse(self, c):
        """For inference, z --> x
            c shape = (B, dim_condit, nsteps_c)
        return:
            x shape = (B, dim_motion, nsteps_x)
        """
        feat_c = self.upsampling_condition(c)
        B, nsteps = feat_c.size(0), feat_c.size(2)

        # add z
        mu  = torch.zeros(B, self.nc_motion_rest, nsteps, device=c.device)
        std = torch.ones(B, self.nc_motion_rest, nsteps, device=c.device)  * self.lambda_std[0]
        x = torch.normal(mean=mu, std=std)
        # x = torch.zeros(B, self.nc_motion_rest, nsteps, device=c.device).normal_()  # (B, 40, 60)

        logdet = torch.zeros(c.size(0), dtype=c.dtype, device=c.device)
        debug(self.opt, f"[Inverse]  c={c.shape}, x={x.shape}, feac_c={feat_c.shape}")

        istd = 1
        for k in reversed(range(self.n_flows)):
            # coupling layer
            nc_motion_half = x.size(1) - x.size(1) // 2
            x1, x2 = x[:, :nc_motion_half, :], x[:, nc_motion_half:, :]
            debug(self.opt, f"[Inverse] k={k}, x1={x1.shape}, x2={x2.shape}", verbose=False)
        
            feat = self.FusionNets[k](x1, feat_c)                                       # (B, 1, dim_output)
            shift, scale = feat[:, 0::2, :],  feat[:, 1::2, :] 
            scale = torch.sigmoid(scale + 2.) + 1e-6     # to make scale > 1, so that log(s) > 0  
            debug(self.opt, f"[Inverse] k={k}, coupling feat shape={feat.shape}, shift={shift.shape}, scale={scale.shape}", verbose=False)

            x2 = x2 / scale - shift
            x = torch.cat([x1, x2], dim=1)
            logdet = - torch.sum(torch.log(scale), dim=[1,2]) + logdet
            debug(self.opt, f"[Inverse] k={k}, coupling concat x={x.shape}, logdet={logdet.shape}", verbose=False)

            # 1 x 1 conv inv
            x, logdet = self.Conv1x1Invs[k].inverse(x, logdet)
            debug(self.opt, f"[Inverse] k={k}, Conv1x1Invs x={x.shape}")
        
            # actnorm
            x, logdet = self.Actnorms[k].inverse(x, logdet)
            debug(self.opt, f"[Inverse] k={k},  actnorm x shape = {x.shape}", verbose=False)

            # add noise 
            if k % self.n_early_every == 0 and k > 0:
                # z = torch.zeros(B, x.size(1), nsteps, device=c.device).normal_()  # (B, 40, 60)

                mu  = torch.zeros(B, x.size(1), nsteps, device=c.device)
                std = torch.ones(B, x.size(1), nsteps, device=c.device) * self.lambda_std[istd]
                z = torch.normal(mean=mu, std=std)
                istd += 1

                x = torch.cat((x, z), dim=1)                                     # (B, 8/40/72/104/136, 25)
                debug(self.opt, f"[Inverse] k={k}, ADD noise + {x.size(1)} channel, x shape = {x.shape}", verbose=False)

        return x.detach(), logdet  # (B, nc_motion, nsteps)
    

    def loss_function_(self, z, logdet, sigma=1.0):
        """z        = list (B, dim_motion, N)
           logdet   = list (B, )
        """

        loss = 0
        loss_dict = {}
        nsteps = z[0].size(2)
        debug(self.opt, f"[Loss] z shape={z[0].shape}, logdet={logdet.shape}")
        assert len(z) == len(self.lambda_std)

        logp = 0
        for std, zn in zip(self.lambda_std, z):
            ilogp = -torch.log(std) - 0.5*zn*zn / (std * std) - 0.5 * torch.log(2 * torch.tensor(np.pi))  # B x nc x nsteps
            logp += torch.sum(ilogp, dim=[1,2])  / (2*sigma * sigma)              # (B, )

        # logp = - 0.5*z*z  - 0.5 * torch.log(2 * torch.tensor(np.pi))  # (B, N, dim_motion)
        # logp = torch.sum(logp, dim=[1,2])  / (2*sigma * sigma)              # (B, )

        loss_logp   = - torch.mean(logp) / (np.log(2.0) * nsteps * len(z))
        loss_logdet = - torch.mean(logdet) / (np.log(2.0) * nsteps)
        loss = loss_logp * self.opt.LOSS.lambda_logp + loss_logdet * self.opt.LOSS.lambda_logdet
        
        loss_dict["loss_logp"] =   loss_logp.detach()
        loss_dict["loss_logdet"] = loss_logdet.detach()
        loss_dict["loss_all"] = loss.detach()

        return loss, loss_dict


class FusionNet(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, opt, nc_motion, nc_condit, nc_output):
        super(FusionNet, self).__init__()

        self.opt = opt
        
        self.n_layers = opt.MODEL.n_layer_wave                     # 4
        self.nc_latent = opt.MODEL.dim_latent                      # 256

        kernel_size = opt.MODEL.kernel_size                        # 3
        assert(kernel_size % 2 == 1)
        assert(self.nc_latent % 2 == 0)


        layer_head = nn.Conv1d(nc_motion, self.nc_latent, kernel_size=1)
        layer_head = nn.utils.weight_norm(layer_head, name='weight')
        self.layer_head = layer_head

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        layer_tail = nn.Conv1d(self.nc_latent, nc_output, kernel_size=1)
        layer_tail.weight.data.zero_()
        layer_tail.bias.data.zero_()
        self.layer_tail = layer_tail

        # for 
        layer_cond = nn.Conv1d(nc_condit, 2*self.nc_latent*self.n_layers, 1)
        self.layer_cond = nn.utils.weight_norm(layer_cond, name='weight')
        self.layer_cond = layer_cond

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(self.n_layers):
            dilation = 2 ** i                                       # 1, 2, 4, 8
            padding = (kernel_size * dilation - dilation) // 2      # 1, 2, 4, 8
            in_layer = nn.Conv1d(self.nc_latent,  2*self.nc_latent, kernel_size, dilation=dilation, padding=padding)
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            nc_skip = 2 * self.nc_latent if (i < self.n_layers - 1) else self.nc_latent
            layer_skip = nn.Conv1d(self.nc_latent, nc_skip, kernel_size=1)
            layer_skip = nn.utils.weight_norm(layer_skip, name='weight')
            self.res_skip_layers.append(layer_skip)


    def forward(self, x, c):
        """ forward function
        x shape = (B x nfeat1 x nsteps1)        B x nc_motion_half x 25
        c shape = (B x nfeat2 x nsteps2)        B x nc_condit*nc_motion x 25
        """

        feat_x = self.layer_head(x)          # B x   256 x 60
        feat_c = self.layer_cond(c)          # B x 4*256 x 25
        feat_o = torch.zeros_like(feat_x)    # B x   256 x 25
        debug(self.opt, f"\t[FusionNet] feat_x={feat_x.shape}, feat_c={feat_c.shape}, feat_o={feat_o.shape}")
        
        nc_latent_tensor = torch.IntTensor([self.nc_latent])

        for i in range(self.n_layers):
            c_offset = i * 2 * self.nc_latent
            debug(self.opt, f"\t[FusionNet] layers={i} c_offset={c_offset}")
        

            ifeat_x = self.in_layers[i](feat_x)                           # B x 512 x 25
            ifeat_c = feat_c[:, c_offset:c_offset+2*self.nc_latent, :]    # B x 512 x 25

            debug(self.opt, f"\t[FusionNet] \tifeat_x={ifeat_x.shape} ifeat_c={ifeat_c.shape}")

            ifeat_m = fused_add_tanh_sigmoid_multiply(ifeat_x, ifeat_c, nc_latent_tensor)  # B x 256 x 25 
            debug(self.opt, f"\t[FusionNet] \tifeat_m={ifeat_m.shape}")


            ifeat_m = self.res_skip_layers[i](ifeat_m)    # B x 512 x 25 if not last layer else B x 256 x 25
            debug(self.opt, f"\t[FusionNet] \t res skip ifeat_m={ifeat_m.shape}")

            if i < self.n_layers - 1:
                feat_x = feat_x + ifeat_m[:, :self.nc_latent, :]       # B x 256 x 25
                feat_o = feat_o + ifeat_m[:, self.nc_latent:, :]       # B x 256 x 25
            else:
                feat_o = feat_o + ifeat_m                              # B x 256 x 25
        
        feat_o = self.layer_tail(feat_o)                               # B x nc_motion_half x 60
        debug(self.opt, f"\t[FusionNet] tail feat_o={feat_o.shape}")

        return feat_o

