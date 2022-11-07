import numpy as np
from scipy import signal
from scipy.io import loadmat
import torch


class IIRFilter:
    def __init__(self, coef: str, bandmode: str = 'bandstop') -> None:
        coefmat = loadmat(coef)
        coef = coefmat['filtcoef'][bandmode][0][0]
        self.coef = coef

    def step(self, waveform: np.ndarray, band: str):
        if band == "all":
            return waveform
        coef = self.coef[band][0][0]
        sos = coef["sos"][0][0].copy(order="C")
        g = coef["g"][0][0].copy(order="C")
        waveform = waveform.copy(order="C")
        waveform = np.prod(g) * signal.sosfilt(sos, waveform, axis=0)
        return waveform


def chMap(eeg):
    channelMap = {
        0: (4, 5),
        1: (2, 5),
        2: (0, 4),
        3: (2, 1),
        4: (2, 3),
        5: (3, 4),
        6: (4, 3),
        7: (3, 2),
        8: (3, 0),
        9: (4, 1),
        10: (5, 2),
        11: (5, 4),
        12: (6, 3),
        13: (6, 1),
        14: (7, 0),
        15: (8, 4),
        16: (6, 5),
        17: (8, 5),
        18: (8, 6),
        19: (7, 10),
        20: (6, 9),
        21: (6, 7),
        22: (5, 6),
        23: (5, 8),
        24: (4, 9),
        25: (3, 10),
        26: (3, 8),
        27: (4, 7),
        28: (3, 6),
        29: (2, 7),
        30: (2, 9),
        31: (0, 6),
    }
    mapwave = np.zeros((eeg.shape[0], eeg.shape[1], 9, 11))
    for chidx in range(eeg.shape[-1]):
        chmap = channelMap[chidx]
        mapwave[:, :, chmap[0], chmap[1]] = eeg[:, :, chidx]
    return mapwave


def networkChMap(eeg):
    channelMap = {
        0: (4, 5),
        1: (2, 5),
        2: (0, 4),
        3: (2, 1),
        4: (2, 3),
        5: (3, 4),
        6: (4, 3),
        7: (3, 2),
        8: (3, 0),
        9: (4, 1),
        10: (5, 2),
        11: (5, 4),
        12: (6, 3),
        13: (6, 1),
        14: (7, 0),
        15: (8, 4),
        16: (6, 5),
        17: (8, 5),
        18: (8, 6),
        19: (7, 10),
        20: (6, 9),
        21: (6, 7),
        22: (5, 6),
        23: (5, 8),
        24: (4, 9),
        25: (3, 10),
        26: (3, 8),
        27: (4, 7),
        28: (3, 6),
        29: (2, 7),
        30: (2, 9),
        31: (0, 6),
    }
    mapwave = torch.zeros(
        (eeg.shape[0], eeg.shape[1], eeg.shape[2], 9, 11), device=eeg.device
    )
    for chidx in range(eeg.shape[-1]):
        chmap = channelMap[chidx]
        mapwave[:, :, :, chmap[0], chmap[1]] = eeg[:, :, :, chidx]
    return mapwave


class filterBank:
    def __init__(self) -> None:
        self.iir = IIRFilter("./model/filtcoef.mat", "bandpass")

    def __step__(self, eeg: np.ndarray):
        outwave = np.zeros((4, eeg.shape[0], eeg.shape[1]), np.float32)
        outwave[0, :, :] = self.iir.step(eeg, "delta")
        outwave[1, :, :] = self.iir.step(eeg, "theta")
        outwave[2, :, :] = self.iir.step(eeg, "alpha")
        outwave[3, :, :] = self.iir.step(eeg, "beta")
        return outwave

