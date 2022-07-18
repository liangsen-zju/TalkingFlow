import os
import cv2
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
plt.rcParams['axes.facecolor'] = 'black'


def save_animation_with_gt(data_motion_pt, data_motion_gt, path_save, suffix, w=128*5, h=72*5, dpi=80, maxsave=3):
    """ convert data_motion_pt to .mp4 animation
    data_motion_pt,    shape [B, N, 68, 2]
    data_motion_gt, shape [B, N, 68, 2]
    """
    fps = 30
    B, *_ = data_motion_pt.shape
    B = min(B, maxsave)
    
    for i in range(B):
        idata_pt = data_motion_pt[i].reshape(-1, 68, 2)
        idata_gt = data_motion_gt[i].reshape(-1, 68, 2)

        images_pt = landmark_to_images(idata_pt, w=w, h=h, dpi=dpi, norm=True, prex="PT")  
        images_gt = landmark_to_images(idata_gt, w=w, h=h, dpi=dpi, norm=True, prex="GT")  

        duration = idata_pt.shape[0] / fps
        # animation = mpy.VideoClip(lambda t: images_pt[int(fps * t + 0.5) - 1,...], duration= duration )
        animation_pt = mpy.VideoClip(lambda t: images_pt[int(fps * t + 0.5),...], duration= duration)
        animation_gt = mpy.VideoClip(lambda t: images_gt[int(fps * t + 0.5),...], duration= duration)

        # mixture
        animation = mpy.clips_array([[animation_pt, animation_gt]])


        ipath_save = path_save.joinpath(f"{suffix}_{i:02d}.mp4")
        animation.write_videofile(str(ipath_save), fps=fps, logger=None)
        

def landmarks_to_video_with_gt(data_motion_pt, data_motion_gt, path_video, suffix, w=128*5, h=72*5, dpi=80, maxsave=3, fps=30):

    """ convert data_motion_pt to .mp4 animation
    data_motion_pt,    shape [B, N, 68, 2]
    data_motion_gt, shape [B, N, 68, 2]
    """


    B, *_ = data_motion_pt.shape
    B = min(B, maxsave)
    
    for i in range(B):
        # ffmpeg writer  
        ipath_save = path_video.joinpath(f"{suffix}_{i:02d}.mp4")
        if int(cv2.__version__[0]) < 3:
            iwriter = cv2.VideoWriter(str(ipath_save), cv2.cv.CV_FOURCC(*'mp4v'), fps, (w, h), True)
        else:
            iwriter = cv2.VideoWriter(str(ipath_save), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)


        idata_pt = data_motion_pt[i].reshape(-1, 68, 2)
        idata_gt = data_motion_gt[i].reshape(-1, 68, 2)

        images_pt = landmark_to_images(idata_pt, w=w, h=h, dpi=dpi, norm=True, prex="PT")  # (N, H, W, 3)
        images_gt = landmark_to_images(idata_gt, w=w, h=h, dpi=dpi, norm=True, prex="GT")  # (N, H, W, 3)

        # concate 
        image = np.stack([images_pt, images_gt], axis=2)    # (nsteps, H, 2W, 3)
        
        # save to writer
        for j in range(image.shape[0]):
            iwriter.write(image[j].astype(np.uint8)[:, :, ::-1])
        iwriter.release()

        # add audio 
        # path_new = path_video.parent.joinpath(f"{path_video.stem}_wt_audio.mp4")
        # cmd = f"ffmpeg -y -i {str(path_video)} -i {str(path_audio)} -map 0:v -vcodec libx264 -map 1:a -c:v copy -shortest -loglevel quiet {str(path_new)}" 
        # os.system(cmd)


def landmarks_to_video(data_motion, path_video, path_audio, fps=30, w=256, h=256, dpi=150):
    """ convert data_motion to .mp4 animation
    data_motion, shape [nsteps, 136]
    """

    # ffmpeg writer  
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(str(path_video), cv2.cv.CV_FOURCC(*'mp4v'), fps, (w, h), True)
    else:
        writer = cv2.VideoWriter(str(path_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

    # landmarks to images 
    data_motion = data_motion.copy().reshape(-1, 68, 2)                                   # nsteps x 68 x 2
    image = landmark_to_images(data_motion, w=w, h=h, dpi=dpi, norm=True, prex="PT")      # (nsteps, H, W, 3)

    # save to writer
    for j in range(image.shape[0]):
        writer.write(image[j].astype(np.uint8)[:, :, ::-1])
    writer.release()

    # add audio
    path_new = path_video.parent.joinpath(f"{path_video.stem}_wt_audio.mp4")
    cmd = f"ffmpeg -y -i {str(path_video)} -i {str(path_audio)} -map 0:v -vcodec libx264 -map 1:a -c:v copy -shortest -loglevel quiet {str(path_new)}" 
    os.system(cmd)
    


def landmark_normalize(ldmk):
    # ldmk shape = (68, 2)
    
    pos_mean = np.mean(ldmk, axis=0)  # (2, )
    pos_max = np.max(ldmk, axis=0)    # (2, )
    pos_min = np.min(ldmk, axis=0)    # (2, )
    
    side = int(1.5 * np.max(pos_max - pos_min))   # (1, )
    # print(f"landmarks = {ldmk[::10, :]}, \n mean={pos_mean}, max={pos_max}, min={pos_min}, side={side}")

    tran = pos_mean - 0.5 * side             # (2,   )
    ldmk_tran = ldmk - tran[np.newaxis, :]   # (68, 2)
    ldmk_norm = ldmk_tran / side             # (68, 2)
    # print(f"ldmk_tran = {ldmk_tran[::10, :]}, \n ldmk_tran = {ldmk_norm[::10, :]}")

    return ldmk_norm, pos_mean, side

def landmark_normalize_by_param(ldmk, params):
    # ldmk shape = (B, 68, 2)
    # params shape = (B, 3)

    pos_mean = params[:, 0:2 ]              # (B, 2)
    pos_mean = pos_mean[:, np.newaxis, :]   # (B, 1, 2)
    side = params[:, 2:]                    # (B, 2)
    side = side[:, np.newaxis, :]           # (B, 1, 2)

    tran = pos_mean - 0.5 * side            # (B, 1, 2)
    ldmk_norm = (ldmk - tran) / side        # (B, 68, 2)

    return ldmk_norm

def landmark_to_images(ldmks, w=256, h=256, dpi=100, norm=True, prex=""):
    # landmarks shape = (nbatch, 68, 2)
    nframe = ldmks.shape[0]

    v_images = []
    for i in range(nframe):
        ildmk = ldmks[i, :, :]
        if prex is not None:
            text = f"{prex}-frame-{i:04d}"
        else:
            text = None

        v_images.append(landmark_to_image(ildmk, w, h, dpi, norm, text))

    return np.asarray(v_images)

def landmark_to_image(ildmk, w=256, h=256, dpi=100, norm=True, text=None):
    # landmarks shape = (68, 2)

    if norm:
        ildmk[:, 0] = w - ildmk[:, 0] * w
        ildmk[:, 1] = h - ildmk[:, 1] * h
        
    else:
        # ildmk[:, 0] = w - ildmk[:, 0]
        ildmk[:, 1] = h - ildmk[:, 1]

    if isinstance(ildmk, torch.Tensor):
        ildmk = ildmk.cpu().numpy()

    # plot 
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor('w')

    ax = fig.add_subplot(1,1,1)
    # ax.imshow(np.zeros(shape=(w, h)))

    ax.set_facecolor('m')
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ax.scatter(ildmk[:, 0], ildmk[:, 1],  marker='o', s=5, color="#808080",alpha=1)
    ax.scatter(ildmk[ 0:17,0], ildmk[ 0:17,1],  marker='o', s=3, color="#008B8B",alpha=1)
    ax.scatter(ildmk[17:22,0], ildmk[17:22,1],  marker='o', s=3, color="#F4A460",alpha=1)
    ax.scatter(ildmk[22:27,0], ildmk[22:27,1],  marker='o', s=3, color="#F4A460",alpha=1)
    ax.scatter(ildmk[27:36,0], ildmk[27:36,1],  marker='o', s=3, color="blue",alpha=1)
    ax.scatter(ildmk[36:48,0], ildmk[36:48,1],  marker='o', s=3, color="#DC143C",alpha=1)
    ax.scatter(ildmk[48:60,0], ildmk[48:60,1],  marker='o', s=3, color="#DC143C",alpha=1)
    ax.scatter(ildmk[60:68,0], ildmk[60:68,1],  marker='o', s=3, color="#FF1493",alpha=1)

    #chin
    ax.plot(ildmk[ 0:17,0], ildmk[ 0:17,1], marker='', markersize=1, linestyle='-', color='#008B8B', lw=2, alpha=1)
    
    #left and right eyebrow
    ax.plot(ildmk[17:22,0], ildmk[17:22,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
    ax.plot(ildmk[22:27,0], ildmk[22:27,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
    
    #nose
    ax.plot(ildmk[27:31,0], ildmk[27:31,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)
    ax.plot(ildmk[31:36,0], ildmk[31:36,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)

    #left and right eye
    ax.plot(ildmk[36:42,0], ildmk[36:42,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[42:48,0], ildmk[42:48,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)

    ax.plot(ildmk[[41,36],0], ildmk[[41,36],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[[47,42],0], ildmk[[47,42],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    
    #outer and inner lip
    ax.plot(ildmk[48:60,0], ildmk[48:60,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[60:68,0], ildmk[60:68,1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

    ax.plot(ildmk[[59,48],0], ildmk[[59,48],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[[67,60],0], ildmk[[67,60],1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')

    if text is not None:
        ax.text(2, 2, text)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape((h, w, 3))
    image = np.asarray(buffer)
    # print(f"w,h={w},{h}, buffer shape = {buffer.shape}, dtype = {buffer.dtype}, class = {(buffer)}")

    plt.close(fig)

    return image

def landmark_to_mask(ildmk, w=256, h=256, dpi=100, norm=True):
    # landmarks shape = (68, 2)
    # print(f"landmark data shape = {ildmk.shape}")
    if norm:
        ildmk = ildmk * w

    # plot 
    fig = plt.figure(figsize=(h/dpi, w/dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    # ax.imshow(np.ones(shape=(w, h)))
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    #chin
    ax.plot(ildmk[ 0:17,0], ildmk[ 0:17,1], marker='', markersize=1, linestyle='-', color='green', lw=2)
    #left and right eyebrow
    ax.plot(ildmk[17:22,0], ildmk[17:22,1], marker='', markersize=1, linestyle='-', color='orange', lw=2)
    ax.plot(ildmk[22:27,0], ildmk[22:27,1], marker='', markersize=1, linestyle='-', color='orange', lw=2)
    #nose
    ax.plot(ildmk[27:31,0], ildmk[27:31,1], marker='', markersize=1, linestyle='-', color='blue', lw=2)
    ax.plot(ildmk[31:36,0], ildmk[31:36,1], marker='', markersize=1, linestyle='-', color='blue', lw=2)
    #left and right eye
    ax.plot(ildmk[36:42,0], ildmk[36:42,1], marker='', markersize=1, linestyle='-', color='red', lw=2)
    ax.plot(ildmk[42:48,0], ildmk[42:48,1], marker='', markersize=1, linestyle='-', color='red', lw=2)
    #outer and inner lip
    ax.plot(ildmk[48:60,0], ildmk[48:60,1], marker='', markersize=1, linestyle='-', color='purple', lw=2)
    ax.plot(ildmk[60:68,0], ildmk[60:68,1], marker='', markersize=1, linestyle='-', color='pink', lw=2) 
    ax.axis('off')


    fig.canvas.draw()


    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # buffer.shape = (w, h, 4)

    image = buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image







