import os
import time
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class ObamaIterDataset(Dataset):

    def __init__(self, opt, phase="train"):
        """
        phase = "train | valid | test "
        """
        super(ObamaIterDataset, self).__init__()

        # path
        self.opt = opt
        self.phase = phase
        path_data= Path(opt.DATASET.root).joinpath(f"data_{phase}.npz")
        
        print(f"\n [{self.phase}] load data ... --> {path_data}")
        self.annodb = np.load(path_data, allow_pickle=True)['data'].item()              # dict vid: {"data_motion", "data_speech", "nframe", "W", "H", ""}

        self.num_pid = len(self.annodb)
        self.pids = list(self.annodb.keys())
        self.annodb_accums = self.count_annotations()
        # print(self.annodb_accums)

        
        self.fps = 30
        self.nwind  = eval(f"opt.DATASET.nwind_second_{phase}")
        self.mean_segmt_frame_rato = self.get_mean_segmt_frame_rato()

        self.nsteps_frame = self.fps * self.nwind
        self.nsteps_segmt = int(self.mean_segmt_frame_rato * self.nsteps_frame + 0.5)
        print(f"len(annodb)={len(self.annodb)}, nsteps_frame={self.nsteps_frame}, nsteps_segmt={self.nsteps_segmt}, mean_segmt_frame_rato={self.mean_segmt_frame_rato}")
        print(f"total frames ={self.annodb_accums[-1]}, iter len ={self.__len__()}")


    def __len__(self):
        return self.annodb_accums[-1] // int(self.nsteps_frame * 0.2)


    def __getitem__(self, idx):
        np.random.seed(int(idx + time.time()))
        vid = self.pids[idx % self.num_pid]

        # random select idx 
        data = self.annodb[vid]
        data_motion = data["data_motion"]
        data_speech = data["data_speech"]

        nframe = data_motion.shape[0]
        nsegmt = data_speech.shape[0]

        # random select beg:end frame
        beg_frame = np.random.randint(nframe - self.nsteps_frame - 2) + 1
        end_frame = beg_frame + self.nsteps_frame

        beg_segmt = int(beg_frame * self.mean_segmt_frame_rato + 0.5)
        end_segmt = beg_segmt + self.nsteps_segmt
        # print(f"nframe-{self.nsteps_frame:04d} =[{beg_frame:04d},{end_frame:04d})\tnsegmet-{self.nsteps_segmt:04d}=[{beg_segmt:04d},{end_segmt:04d}), vid={vid}")

        result = {
            "data": data_motion[beg_frame:end_frame, :],                             # (nsteps_frame, dim_motion)
            "cond": data_speech[beg_segmt:end_segmt, :],                             # (nsteps_frame, dim_speech)
            "vid": vid,
            "frame_interval": [beg_frame, end_frame]
        } 

        if "data_pose" in data.keys():
            result["pose"] = data["data_pose"][beg_frame:end_frame, :]

        return  result
            
            
    def count_annotations(self):
        acc = 0
        annodb_accums = np.zeros(self.num_pid + 1, dtype=int)  # the first element set 0

        for i, anno_dict in enumerate(self.annodb.values()):
            acc += anno_dict["data_motion"].shape[0] 
            annodb_accums[i+1] =  acc          # 2

        return annodb_accums


    def get_mean_segmt_frame_rato(self):
        ratio = []
        for vid, anno_dict in self.annodb.items():
            ratio = anno_dict["data_speech"].shape[0] / anno_dict["data_motion"].shape[0] 
        return np.mean(ratio)



