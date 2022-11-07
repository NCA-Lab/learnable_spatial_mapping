import torch
import os
from model.utils import *
import model.Utils_Bashivan as utilb
from torch import Tensor
from scipy import stats


class imageDataset(torch.utils.data.Dataset):
    """Some Information about audioDataset"""

    def __init__(
        self,
        imageDir,
        winlen,
        mode="train",
        sigma=0.0001,
        nfft=128,
        isUseFilterBank=True,
        isChannelMapping=True,
    ):
        super(imageDataset, self).__init__()
        self.rootdir = imageDir + mode + "/"
        self.classnames = os.listdir(self.rootdir)
        self.winlen = winlen
        self.mode = mode
        self.nfft = nfft
        self.datasetLength = []
        self.eachClassDir = []
        for classname in self.classnames:
            classdir = os.path.join(self.rootdir, classname)
            ls = os.listdir(classdir)
            length = len(ls.copy())
            self.datasetLength.append(length)
            self.eachClassDir.append(ls.copy())
        self.sigma = sigma
        self.mode = mode
        self.isUseFilterBank = isUseFilterBank
        self.isChannelMapping = isChannelMapping
        self.filterBank = filterBank()
        self.bandMap = {
            "delta": 0,
            "theta": 1,
            "alpha": 2,
            "beta": 3,
        }

    def __getitem__(self, index):
        if (self.mode == 'train') & self.isUseFilterBank:
            augBand = index - index // 4 * 4
            augBand /= 4
            index //= 4
            # augBand = torch.randint(0, 4, (1,)) / 4
        previousClassIdx = 0
        nextClassIdx = 0
        breakFlag = False
        for i in range(len(self.datasetLength)):
            nextClassIdx += self.datasetLength[i]
            if (previousClassIdx <= index) & (index < nextClassIdx):
                indexToRead = index - previousClassIdx
                thisClassDir = self.eachClassDir[i]
                thisClassName = self.classnames[i]
                thisImageDir = thisClassDir[indexToRead]
                imageDirToRead = self.rootdir + thisClassName + "/" + thisImageDir
                image = loadmat(imageDirToRead)
                eeg = image["eegslice"]
                attended_LR = i
                breakFlag = True
            previousClassIdx = nextClassIdx
            if breakFlag:
                break
        eeg = stats.zscore(eeg, ddof=1, axis=None)
        eeg = self.filterBank.__step__(eeg)
        # Band * Time * Channel
        eeg = np.swapaxes(eeg, 0, 1)
        if self.isChannelMapping:
            eeg = chMap(eeg)
        if (self.mode == 'train') & self.isUseFilterBank:
            augBand = self.__getAugBand__(augBand)
            eeg = self.__randomRemoveFreqBand__(eeg, augBand)
        eeg = torch.from_numpy(eeg).float()
        subjectId = int(thisImageDir[1:].split('_')[0])
        trialID = int(thisImageDir[1:].split('_')[2])
        sliceID = int(thisImageDir[1:].split('_')[4])
        eeg = eeg.permute((1, 0, 2))

        return (subjectId, trialID, sliceID, eeg, attended_LR)

    def __len__(self):
        if (self.mode == 'train') & self.isUseFilterBank:
            return sum(self.datasetLength) * 4
        else:
            return sum(self.datasetLength)
        # return sum(self.datasetLength)

    def getClassWeight(self):
        weight = Tensor(self.datasetLength)
        return weight

    def getClassName(self):
        name = self.classnames
        return name

    def getClassPath(self):
        path = self.eachClassDir
        return path

    def getNumClass(self):
        return len(self.classnames)

    def __randomRemoveFreqBand__(self, x: np.ndarray, band: str):
        if not band == "all":
            bandIdx = self.bandMap[band]
            x[:, bandIdx] = (
                np.random.standard_normal((x.shape[0],) + (x.shape[2:])) * 0.01
            )
        return x

    def __getAugBand__(self, augRand):
        if augRand < 0.25:
            band = "delta"
        elif augRand < 0.5:
            band = "alpha"
        elif augRand < 0.75:
            band = "theta"
        else:
            band = "all"
        return band

    def __randomAddGuassianNoise__(self, x: np.ndarray, sigma: float):
        # input: Band * Time * Channel
        wgn = np.random.standard_normal(x.shape) * sigma
        x = x + wgn
        return x

    def __estimateSignalPower__(self, x: np.ndarray, dim=-1):
        power = np.sqrt(np.sum(np.square(x), axis=dim) / x.shape[dim])
        return power

    def __estimateSigma__(self, power, snr):
        sigma = power / (10 ** (snr / 20))
        return sigma
