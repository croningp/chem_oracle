import glob
from functools import reduce
from operator import add
from os import path

import numpy as np
import tensorflow as tf
from nmr_analyze.nmr_analysis import NMRSpectrum

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def process_nmr(nmr_path, length=None, normalize=True):
    r = (
        NMRSpectrum(nmr_path)
        .crop(0, 12, inplace=True)
        #.remove_peak(2.5, rel_height=0.9999, inplace=True, cut=False)
        #.autophase(inplace=True)
        .cut(1.3, 3.65, inplace=True)  # remove DMSO peak region
        .normalize(inplace=True)
    )
    if normalize:
        r.normalize(inplace=True)
    if length is not None:
        interpolator = interp1d(r.xscale, r.spectrum)
        r.xscale = np.linspace(r.xscale[0], r.xscale[-1], length)
        r.spectrum = interpolator(r.xscale)
        r.length = length
    return r


class NMRDataset:
    def __init__(self, dirs, adjust_length=True, dtype="float32", normalize=True, target_length=None):
        self.dirs = dirs
        self.spectra = [process_nmr(d) for d in self.dirs]
        if adjust_length:
            self.min_length = target_length or min(len(s) for s in self.spectra)
            self.spectra = [
                process_nmr(d, self.min_length, normalize=normalize) for d in self.dirs
            ]
        self.matrix = np.vstack(
            [s.spectrum.real[: self.min_length] for s in self.spectra]
        ).astype(dtype, casting="same_kind")
        if normalize:
            self.matrix / self.matrix.max(axis=1)[:, np.newaxis]

    def __iter__(self):
        return self.spectra.__iter__()

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # return from matrix
            return self.matrix[idx]
        else:
            # return from spectra
            return self.spectra[idx]

TARGET_LENGTH = 3921
DATA_FOLDER = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data2"
MODEL_HOME = "/home/group/Code/NMRModel"

REAGENT_DIRS = {
    int(path.basename(p).split("_")[1]): p
    for p in glob.glob(path.join(DATA_FOLDER, "reagents", "*_1H"))
}

REAGENT_SPECTRA = NMRDataset([REAGENT_DIRS[i] for i in sorted(REAGENT_DIRS)], target_length=TARGET_LENGTH)

MODELS = {
    path.basename(model_path): tf.keras.models.load_model(model_path)
    for model_path in glob.glob(path.join(MODEL_HOME, "*.tf"))
}


def test_point(*spectra: NMRSpectrum):
    return np.vstack([s.spectrum.real for s in spectra])[np.newaxis, ...]


def nmr_process(folder: str, model: tf.keras.Model) -> bool:
    rxn_spec = process_nmr(folder, length=TARGET_LENGTH)
    reagents = [p for p in path.basename(folder).split("_")[1].split("-")]
    sms = reduce(add, [REAGENT_SPECTRA[int(i)] for i in reagents]).normalize()
    prediction = model.predict(test_point(rxn_spec, sms))[0, 0]
    return prediction > 0.5 and 1.0 or 0.0
