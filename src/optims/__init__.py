import torch
import torch.nn as nn
from torch.optim import lr_scheduler

def get_optimizer(net, opt):

    optim = None

    if opt.TRAIN.optimizer == "Adam":
        optim = torch.optim.Adam( 
            filter(lambda p: p.requires_grad, net.parameters()), 
            lr=opt.TRAIN.lr, 
            betas=(opt.TRAIN.beta1, opt.TRAIN.beta2), \
            weight_decay=opt.TRAIN.weight_decay)

    elif opt.TRAIN.optimizer == "SGD":
        optim = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()), 
            lr=opt.TRAIN.lr, 
            momentum=opt.TRAIN.momentum, 
            weight_decay=opt.TRAIN.weight_decay, 
            nesterov=opt.TRAIN.nesterov)

    else:
        raise NotImplementedError(f"opt.TRAIN.optimizer ({opt.TRAIN.optimizer}) NOT implemented!! ")


    return optim

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.TRAIN.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.TRAIN.epoch_begin - opt.TRAIN.epoch_end) / float(opt.TRAIN.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
    elif opt.TRAIN.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.TRAIN.lr_decay_iters, gamma=0.1)

    elif opt.TRAIN.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.TRAIN.lr_milestones, gamma=0.1)

    elif opt.TRAIN.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  mode='min',     # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
                                                  factor=0.5,     # new_lr = lr * factor. Default: 0.1., Factor by which the learning rate will be reduced.
                                                  threshold=1e-4, # Threshold for measuring the new optimum
                                                  min_lr=1e-6,    # A lower bound on the learning rate 
                                                  patience=5,     # Number of epochs with no improvement after which learning rate will be reduced
                                                  )
    elif opt.TRAIN.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.TRAIN.epoch_end, eta_min=1e-6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.TRAIN.lr_policy)
    
    
    return scheduler
