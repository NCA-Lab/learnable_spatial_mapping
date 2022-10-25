from email.mime import image
from datetime import datetime
from re import I
import shutil
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch import optim
import os
from torch import Tensor
from sklearn.metrics import confusion_matrix
from memory_profiler import profile
from scipy.io import loadmat, savemat
from scipy import signal
import numpy as np
from thop import profile as thop_prof
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tempfile
import torch.multiprocessing as mp
import model.Utils_Bashivan as utilb
from torch.utils.tensorboard import SummaryWriter
import math as m

# from topo_reserved import *
from model.dataset import *
from model.utils import *
from model.attention import *

from warmupoptimizer import NoamOpt as warmUpOptimizer

np.random.seed(123)
torch.cuda.manual_seed_all(123)
torch.manual_seed(123)


class deepNetwork(nn.Module):
    """Some Information about deepNetwork"""

    def __init__(
        self,
        input,
        numConvFilter,
        winlen,
        numFCNeurons,
        numGroup=None,
        enableMaxPool=True,
        maxPoolInt=2,
        maxPollStride=2,
        imageSize=(9, 11),
        imageKernelSize=3,
        device='cpu',
    ):
        super(deepNetwork, self).__init__()

        self.numconv = len(numConvFilter)
        self.numfc = len(numFCNeurons)
        if numGroup is None:
            self.numGroup = numFCNeurons[-1]
        else:
            self.numGroup = numGroup

        input = input.to(device)
        print(input.shape)
        self.freqAttention = bandAttention(input, 1, device=device).to(device)
        input = self.freqAttention(input)
        # batch * band * frame * channel
        self.channelAttention = channelAttention(
            input, imageKernelSize, imageSize, device=device
        ).to(device)
        input = self.channelAttention(input)

        self.convnet = nn.ModuleList()
        maxpoolIdx = 0
        print(input.shape)
        # (batch * frame ) * band * c1 * c2
        for convLayerIdx in range(self.numconv):
            # (Batch * Time) * Band * Chan1 * Chan2
            inputChannel = input.shape[1]
            inputHeight = input.shape[2]
            inputWidth = input.shape[3]
            # convolutionKernel = (inputHeight, inputWidth)
            convolutionKernel = tuple(
                [m.ceil(kernel / (2 ** maxpoolIdx)) for kernel in imageKernelSize]
            )
            conv = nn.Conv3d(
                inputChannel,
                numConvFilter[convLayerIdx],
                convolutionKernel,
                padding='same',
            )
            relu = nn.ReLU()
            bn = nn.BatchNorm3d(numConvFilter[convLayerIdx])
            dp = nn.Dropout()
            net = nn.Sequential(conv, relu, bn, dp)
            net = net.to(device)
            input = net(input)
            self.convnet.add_module("conv%02d" % (convLayerIdx + 1), net)
            print(input.shape)

        # Average pooling
        avgpKernel1 = input.shape[2]
        avgpKernel2 = 1
        avgpKernel3 = 1
        net = nn.AvgPool3d((avgpKernel1, avgpKernel2, avgpKernel3))
        self.add_module("avgp", net)
        net = net.to(device)
        input = net(input)
        # Average pooling across time dimension
        print(input.shape)

        net = nn.Flatten()
        net = net.to(device)
        input = net(input)
        self.add_module("channelFlatten", net)
        print(input.shape)

        # Fully Connect
        input = input.squeeze()
        self.fcnet = nn.ModuleList()
        for fcLayerIdx in range(self.numfc):
            inputChannel = input.shape[1]
            outputChannel = numFCNeurons[fcLayerIdx]
            fc = nn.Linear(inputChannel, outputChannel)
            dp = nn.Dropout()
            net = nn.Sequential(fc, dp)
            net = net.to(device)
            input = net(input)
            self.fcnet.add_module("fc%02d" % (fcLayerIdx + 1), net)
            print(input.shape)


    def forward(self, input):
        input = self.freqAttention(input)
        input = self.channelAttention(input)
        for model in iter(self.convnet):
            input = model(input)

        # Average pooling
        # Batch * Time * Feature * Chan1' * Chan2'
        input = self.avgp(input)
        input = self.channelFlatten(input)
        # Fully Connect
        input = input.squeeze()
        for model in iter(self.fcnet):
            input = model(input)

        return input


# @profile
def run_dist(rank, world_size, tb_filename="./runs/"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12351'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    winlen = 2
    cvAccu = torch.zeros((5, 11))
    for l in range(1, 2):
        l2factor = 1e-4
        for cv in range(1, 6):
            print(f"Cross validation on l2factor {l2factor} fold {cv}")

            # Please specify which dataset to use. You can store the dataset to another path and change the source directory here.

            # srcdir = "./nju/CV_%02d/" % (cv)
            # srcdir = "./kul/CV_%02d/" % (cv)
            target_data_folder = srcdir + "decision_win_%02.1fs/" % (winlen)
            dataset = imageDataset(
                target_data_folder,
                winlen,
                mode="train",
                isChannelMapping=False,
                isUseFilterBank=False,
            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=True,
                num_workers=2,
                prefetch_factor=8,
            )
            valdataset = imageDataset(
                target_data_folder, winlen, mode="val", isChannelMapping=False,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(valdataset)
            valdataloader = torch.utils.data.DataLoader(
                valdataset,
                sampler=val_sampler,
                batch_size=64,
                shuffle=(val_sampler is None),
                drop_last=True,
                num_workers=2,
                prefetch_factor=8,
            )
            testdataset = imageDataset(
                target_data_folder, winlen, mode="test", isChannelMapping=False,
            )
            testsampler = torch.utils.data.distributed.DistributedSampler(testdataset)
            testdataloader = torch.utils.data.DataLoader(
                valdataset,
                batch_size=64,
                shuffle=(testsampler is None),
                drop_last=True,
                num_workers=2,
                sampler=testsampler,
                prefetch_factor=8,
            )
            className = dataset.getClassName()
            print(className)

            epochVerbose = True

            numCategories = dataset.getNumClass()
            print(f"Dataset class length is {dataset.getClassWeight()}")
            print(f"Dataset class name is {dataset.getClassName()}")
            print(f"Validation dataset length is {valdataset.getClassWeight()}")
            print(f"Testing dataset length is {testdataset.getClassWeight()}")

            # You can change the hyperparameters of the network here

            numConvolutinFilter = [8, 8, 12]
            imageSize = (12, 12)
            # imageSize = imageSizePool[0]
            imageKernelSize = (13, 3, 3)
            numFCNeurons = [256, numCategories]

            learningRate = 1e-4
            L2Factor = l2factor
            numEpoch = 120
            lrDropDownFactor = 0.5
            loadFlag = False
            patience = 10
            cooldown = 0

            (s, t, slice, eeg, label) = iter(dataloader).next()
            net = deepNetwork(
                eeg,
                numConvFilter=numConvolutinFilter,
                winlen=winlen,
                numFCNeurons=numFCNeurons,
                numGroup=numCategories,
                enableMaxPool=False,
                imageSize=imageSize,
                imageKernelSize=imageKernelSize,
                device=rank,
            )
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(rank)
            CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
            if rank == 0:
                torch.save(net.state_dict(), CHECKPOINT_PATH)
            dist.barrier()
            net.load_state_dict(
                torch.load(CHECKPOINT_PATH, map_location='cuda:{}'.format(rank))
            )
            net = net.to(rank)
            eeg = eeg.to(rank)
            net = DDP(net, device_ids=[rank])

            trainWeight = dataset.getClassWeight()
            trainWeight = 1 / (trainWeight + 1e-8)
            trainWeight = trainWeight.to(rank)
            valWeight = valdataset.getClassWeight()
            valWeight = 1 / (valWeight + 1e-8)
            valWeight = valWeight.to(rank)
            testWeight = testdataset.getClassWeight()
            testWeight = 1 / (testWeight + 1e-8)
            testWeight = testWeight.to(rank)
            lossfn = nn.CrossEntropyLoss(weight=trainWeight, reduction="mean")

            weight_p, bias_p = [], []
            aweight_p, abias_p = [], []
            for model_name, model in net.named_children():
                if rank == 0:
                    print(model)
                if model_name in ["freqAttention", "chAttention"]:
                    for name, p in model.named_parameters():
                        if "bias" in name:
                            abias_p += [p]
                        else:
                            aweight_p += [p]
                else:
                    for name, p in model.named_parameters():
                        if "bias" in name:
                            bias_p += [p]
                        else:
                            weight_p += [p]

            optimizer = optim.Adam(
                [
                    {
                        "params": weight_p,
                        "weight_decay": L2Factor,
                        "lr": learningRate,
                        "group_name": "default",
                    },
                    {
                        "params": bias_p,
                        "weight_decay": 0,
                        "lr": learningRate,
                        "group_name": "default",
                    },
                    {
                        "params": aweight_p,
                        "weight_decay": L2Factor,
                        "lr": learningRate,
                        "group_name": "attention",
                    },
                    {
                        "params": abias_p,
                        "weight_decay": 0,
                        "lr": learningRate,
                        "group_name": "attention",
                    },
                ],
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=lrDropDownFactor,
                patience=patience,
                cooldown=cooldown,
                min_lr=1e-6,
                verbose=True,
                threshold=0.015,
                threshold_mode="abs",
            )
            best_accu = 0.0
            if rank == 0:
                os.mkdir(tb_filename + "CV%02d" % cv)
            dist.barrier()
            writer = SummaryWriter(log_dir=tb_filename + "CV%02d" % cv)
            writer.add_scalar("LR", learningRate)
            writer.add_scalar("L2", L2Factor)
            for epidx in range(numEpoch):
                net.train()
                train_sampler.set_epoch(epidx)
                val_sampler.set_epoch(epidx)
                testsampler.set_epoch(epidx)
                total_running_loss = torch.zeros((1,), device=rank)
                running_loss = torch.zeros((1,), device=rank)
                total_running_accu = torch.zeros((1,), device=rank)
                running_accu = torch.zeros((1,), device=rank)
                val_accu = torch.zeros((1,), device=rank)
                test_accu = torch.zeros((1,), device=rank)
                subjectAccuracyTest = torch.zeros((30,), device=rank)
                subjectCountTest = torch.zeros((30,), device=rank)
                for i, (s, trial, slice, eeg, label) in enumerate(dataloader):
                    optimizer.zero_grad()
                    eeg = eeg.to(rank)
                    label = label.to(rank)
                    output = net(eeg)
                    loss = lossfn(output, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    total_running_loss += loss.item()

                    classify = torch.max(output, dim=1)[1]
                    accuracy = (classify == label).long()
                    accuracy = accuracy.float()
                    accuracy = accuracy.mean()

                    running_accu += accuracy
                    total_running_accu += accuracy

                writer.add_scalar(
                    "Loss/train", loss.item(), i + epidx * len(dataloader)
                )
                writer.add_scalar(
                    "Accuracy/train", accuracy, i + epidx * len(dataloader)
                )

                # validation
                net.eval()
                with torch.no_grad():
                    for (j, (s, trial, slice, eeg, label)) in enumerate(valdataloader):
                        eeg = eeg.to(rank)
                        label = label.to(rank)
                        output = net(eeg)
                        classify = torch.max(output, dim=1)[1]
                        accuracy = (classify == label).long()
                        accuracy = accuracy.float()
                        accuracy = accuracy.mean()
                        val_accu += accuracy
                        writer.add_scalar(
                            "Accuracy/val", accuracy, j + epidx * len(valdataloader),
                        )
                    # test
                    y_true = torch.ones((1), dtype=torch.int64, device=rank)
                    y_pred = torch.ones((1), dtype=torch.int64, device=rank)
                    for (k, (s, trial, slice, eeg, label)) in enumerate(testdataloader):
                        eeg = eeg.to(rank)
                        label = label.to(rank)
                        output = net(eeg)
                        classify = torch.max(output, dim=1)[1]
                        accuracy = (classify == label).long()
                        y_true = torch.cat((y_true, label), dim=0)
                        y_pred = torch.cat((y_pred, classify), dim=0)
                        accuracy = accuracy.float()
                        for idx in range(len(s)):
                            sIdx = s[idx] - 1
                            subjectAccuracyTest[sIdx] = (
                                subjectAccuracyTest[sIdx] + accuracy[sIdx]
                            )
                            subjectCountTest[sIdx] += 1
                        accuracy = accuracy.mean()

                        test_accu += accuracy
                        writer.add_scalar(
                            "Accuracy/test", accuracy, k + epidx * len(testdataloader),
                        )
                dist.reduce(test_accu, 0)
                dist.reduce(subjectAccuracyTest, 0)
                dist.reduce(subjectCountTest, 0)

                if rank == 0:
                    ytruetmp = torch.zeros(y_true.size(), device=rank, dtype=torch.bool)
                    ypredtmp = torch.zeros(y_true.size(), device=rank, dtype=torch.bool)
                    dist.barrier()
                    for r in range(1, world_size):
                        dist.irecv(ytruetmp, r, tag=r)
                        dist.irecv(ypredtmp, r, tag=r + world_size)
                        y_true = torch.cat([y_true, ytruetmp.int()])
                        y_pred = torch.cat([y_pred, ypredtmp.int()])
                else:
                    dist.barrier()
                    dist.isend(y_true.bool(), 0, tag=rank)
                    dist.isend(y_pred.bool(), 0, tag=rank + world_size)

                if epochVerbose & (rank == 0):
                    y_true = y_true.contiguous()
                    y_pred = y_pred.contiguous()
                    # if epochVerbose:
                    test_accu = test_accu / world_size
                    subjectAccuracyTest = subjectAccuracyTest / subjectCountTest
                    if test_accu / (k + 1) > best_accu:
                        best_accu = test_accu / (k + 1)
                    print(
                        "|[epoch %03d]|training accuracy %.3f|validation  accuracy %.3f|test accuracy %.3f|best accuracy %.3f|"
                        % (
                            epidx + 1,
                            total_running_accu / (i + 1),
                            val_accu / (j + 1),
                            test_accu / (k + 1),
                            best_accu,
                        )
                    )
                    print(
                        "|[epoch %03d]|Subject-wise accuracy:\n" % (epidx + 1),
                        subjectAccuracyTest,
                    )
                    y_true = y_true.cpu()
                    y_pred = y_pred.cpu()
                    conf_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
                    print("Confusion matrix:")
                    print(className)
                    print(conf_mtx)
                scheduler.step(val_accu / (j + 1))
                if rank == 0:
                    cvAccu[cv - 1, epidx % 10] = test_accu / (k + 1)
                    cvAccu[cv - 1, -1] = best_accu
                    print(cvAccu)
                    torch.save(
                        net.state_dict(),
                        tb_filename
                        + "convMap_attention_checkpoint_CV%02d_epoch%03d.pt"
                        % (cv, epidx),
                    )
            dist.barrier()


if __name__ == "__main__":
    print(os.curdir)
    now = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
    filename = "./runs/%s/" % now
    os.mkdir(filename)
    shutil.copy("./convMap+attention.py", filename + "convMap+attention.py")
    mp.spawn(
        run_dist,
        args=(torch.cuda.device_count(), filename),
        nprocs=torch.cuda.device_count(),
        join=True,
    )

