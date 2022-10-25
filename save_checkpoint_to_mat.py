import torch
from scipy.io import savemat, loadmat
import glob
import numpy as np
import model.Utils_Bashivan as utilb


def save_to_mat(checkpoint_file_name):
    d = torch.load(checkpoint_file_name)
    weight = d["module.channelAttention.map.mapnet.0.weight"].squeeze()
    weight = weight.to("cpu").detach()
    weight = weight.numpy()
    process_format(weight, checkpoint_file_name + ".mat")
    del d, weight


def process_batch(dir_name):
    ils = glob.iglob(dir_name + "/*epoch11*.pt")
    for filename in ils:
        save_to_mat(filename)


def process_format(w, filename):
    w = np.sum(w, 0)
    chanlocs = loadmat('emotivChanY+MNI.mat')
    cor = np.zeros((32, 2))
    w = np.expand_dims(w, 1)
    w = np.transpose(w)
    for c in range(len(chanlocs['cor'])):
        cor[c, :] = utilb.azim_proj(chanlocs['cor'][c, :])
    image, x, y = utilb.gen_images(cor, w, 256, edgeless=False)
    image = np.squeeze(image)
    savemat(
        filename,
        {
            "image": image,
            "imagex": x,
            "imagey": y,
            "elecx": cor[:, 0],
            "elecy": cor[:, 1],
        },
        appendmat=False,
    )


if __name__ == "__main__":
    # specify the path contains pytorch module checkpoints
    process_batch("./runs/22-10-12-10:03:12")

