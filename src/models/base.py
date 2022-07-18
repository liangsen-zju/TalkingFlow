import os
import torch
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from pathlib import Path

class Base():
    __metaclass__ = ABCMeta
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call Base.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt, logger):
        """Initialize the Base class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <Base.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.names_loss (str list):          specify the training losses that you want to plot and save.
            -- self.names_model (str list):         specify the images that you want to display and save.
            -- self.names_visual (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.logger = logger

        self.gpu_ids = opt.GPUS
        self.isTrain = opt.isTrain

        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.path_save = Path(opt.OUTPUT_DIR)             # save all the checkpoints to path_save
        self.path_save.mkdir(parents=True, exist_ok=True)
        
        self.names_loss = []
        self.names_model = []
        self.names_visual = []
        
        self.loss_dict = OrderedDict()
        self.data_dict = OrderedDict()

        self.schedulers = []
        self.optimizers = []      
    
        self.logger.info(f"device set to {self.device}")

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.names_model:
            net = getattr(self, 'net' + name)
            net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.names_model:
            net = getattr(self, 'net' + name)
            net.train()

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize> and <test>."""
        pass

    @abstractmethod
    def training(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass
    

    def testing(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
    
    def validation(self):
        """Forward function used in validation time. """
        pass


    def setup(self, opt):
        """Load and print networks; 
        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        # if train
        if self.isTrain:

            # load model for resume
            if opt.TRAIN.resume:
                self.logger.info("SETUP: resume network training")
                load_suffix = f'iter_{opt.TRAIN.load_iter}' if opt.TRAIN.load_iter > 0 else opt.TRAIN.epoch_begin
                path_chekpoints = Path(opt.MODEL.path_checkpoint)
                self.load_networks(load_suffix, path_chekpoints, load_info=True)
                # self.load_networks(load_suffix, self.path_save, load_info=True)

            # load model for pretrain
            elif opt.MODEL.pretrain:
                self.logger.info("SETUP: load pretrain network")
                load_suffix = f'iter_{opt.TRAIN.load_iter}'if opt.TRAIN.load_iter > 0 else opt.MODEL.load_epoch
                path_chekpoints = Path(opt.MODEL.path_checkpoint)
                self.load_networks(load_suffix, path_chekpoints, load_info=False)
            
            # init
            else:
                self.logger.info("SETUP: Init by each network")
    
        # if test or infer
        else:
            self.logger.info("SETUP: TEST or INFER, load pretrain network")
            load_suffix = f'iter_{opt.TRAIN.load_iter}' if opt.TRAIN.load_iter > 0 else opt.MODEL.load_epoch
            path_chekpoints = Path(opt.MODEL.path_checkpoint)
            self.load_networks(load_suffix, path_chekpoints, load_info=False)
        
        # printer networks
        self.print_networks(opt.verbose)


    def update_parameters(self, epoch):
        pass

    def update_learning_rate(self, cur_loss=None):
        """Update learning rates for all the networks; called at the end of every epoch
        `cur_loss` used for learning rate policy 'plateau'
        """
        for scheduler in self.schedulers:
            if self.opt.TRAIN.lr_policy == 'plateau':
                scheduler.step(cur_loss)
            else:
                scheduler.step()

        msg = ""
        for i, optimize in enumerate(self.optimizers):
            lr = optimize.param_groups[0]['lr']
            msg += f"net{self.names_model[i]}={lr:.7f} "

        self.logger.info(f"\t[UPDATE] learning rate:" + msg)

        return lr


    def get_test_visuals(self):
        """Return visualization images. train.py will display these images with 
        visdom, and save the images to a HTML"""
        pass


    def get_train_visuals(self):
        """Return visualization images. train.py will display these images with 
        visdom, and save the images to a HTML"""

        visual_ret = OrderedDict()
        for name in self.names_visual:
            visual_ret[name] = getattr(self, name)
            # get the data from a variable
            if isinstance(visual_ret[name], torch.Tensor):  
                visual_ret[name] = visual_ret[name].detach().cpu().numpy()
                 
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. 
        train.py will print out these errors on console, and save them to a file"""
        
        errors_ret = OrderedDict()
        for name in self.names_loss:
            # print(f"name = ",name)
            # errors_ret[name] = self.loss_dict["loss_" + name ].detach().float().cpu()        # float(...) works for both scalar tensor and float number
            errors_ret[name] = self.loss_dict["loss_" + name ].float().cpu()        # float(...) works for both scalar tensor and float number
            
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        # save parameters
        for name in self.names_model:

            save_path = self.path_save.joinpath("checkpoints", f'{epoch}_net_{name}.pth')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)

        # save training information
        save_path = self.path_save.joinpath("checkpoints", f'{epoch}_net_train_state.pth')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "losses":{},
            "optimizers": {},
            "schedulers": {}
        }
        
        for name in self.names_model:
            optimizer_ = getattr(self, 'optimizer_' + name)
            scheduler_ = getattr(self, 'scheduler_' + name)
            state['optimizers']['optimizer_' + name] = optimizer_.state_dict()
            state['schedulers']['scheduler_' + name] = scheduler_.state_dict()
        
        for name in self.names_loss:
            state["losses"][name] = self.loss_dict["loss_"+name]
        
        torch.save(state, save_path)


    def load_networks(self, load_suffix, load_path, load_info=False):
        """Load all the networks from the disk.
        """
        
        # load parameters
        for name in self.names_model:
            path_checkpoint = load_path.joinpath("checkpoints", f'{load_suffix}_net_{name}.pth')

            load_state = torch.load(str(path_checkpoint), map_location=str(self.device))

            net = getattr(self, 'net' + name)

            # net = net.module if isinstance(net, torch.nn.DataParallel) else net
            # mm = ["encoder_xt", "encoder_xp"]
            if isinstance(net, torch.nn.DataParallel):
                model_state = net.module.state_dict() 

                # state_dict = {}
                # for k,v in load_state.items():
                #     if k in model_state.keys():
                #         print(k, k.split("."))
                #         if k.split(".")[0] not in mm:
                #             state_dict[k] = v 
                # state_dict = {k:v for k,v in load_state.items() if (k in model_state.keys() and k.split(".")[0] not in mm)}
                state_dict = {k:v for k,v in load_state.items() if (k in model_state.keys())}

                # print(state_dict.keys())

                model_state.update(state_dict)
                net.module.load_state_dict(model_state)

            else:
                model_state = net.state_dict() 
                # state_dict = {k:v for k,v in load_state.items() if (k in model_state.keys() and k.split(".")[0] not in mm)}
                state_dict = {k:v for k,v in load_state.items() if (k in model_state.keys())}

                # print(state_dict.keys())

                model_state.update(state_dict)
                net.load_state_dict(model_state)

            self.logger.info('loading the model from %s' % path_checkpoint)
            
            # net.module.set_actnorm_init(True)  # set actnorm 

        if load_info:
            # load training state
            path_state = load_path.joinpath("checkpoints", f'{load_suffix}_net_train_state.pth')
            load_state = torch.load(str(path_state))
            
            for name in self.names_model:
                optimizer_ = getattr(self, 'optimizer_' + name)
                scheduler_ = getattr(self, 'scheduler_' + name)
                optimizer_.load_state_dict(load_state['optimizers']['optimizer_' + name])
                scheduler_.load_state_dict(load_state['schedulers']['scheduler_' + name])

                if self.opt.TRAIN.reset_lr:
                    optimizer_.param_groups[0]['lr'] = self.opt.TRAIN.lr

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        self.logger.info('---------- Networks initialized -------------')
        for name in self.names_model:
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            
            if verbose:
                self.logger.info(net)
            
            self.logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        self.logger.info('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
