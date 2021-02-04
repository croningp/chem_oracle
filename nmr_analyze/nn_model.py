import glob
from functools import reduce
from operator import add
from os import path

import numpy as np
import tensorflow as tf

from nmr_analyze.nmr_analysis import NMRDataset, NMRSpectrum

DEFAULT_XFORM = lambda s: s.crop(0, 12, inplace=True).normalize(inplace=True)


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def process_nmr(nmr: NMRSpectrum, length=None, normalize=True):
    r = (
        nmr.copy().crop(0, 12, inplace=True)
        # .remove_peak(2.5, rel_height=0.9999, inplace=True, cut=False)
        # .autophase(inplace=True)
        # .cut(2.0, 3.0, inplace=True)  # remove DMSO peak region
        .normalize(inplace=True)
    )
    if normalize:
        r.normalize(inplace=True)
    if length is not None:
        r.resize(length, inplace=True)
    return r


TARGET_LENGTH = 2438
DATA_FOLDER = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data2"
MODEL_HOME = "/mnt/orkney1/NMRModel"

REAGENT_DIRS = {
    int(path.basename(p).split("_")[1]): p
    for p in glob.glob(path.join(DATA_FOLDER, "reagents", "*_1H"))
}

REAGENT_SPECTRA = NMRDataset(
    [REAGENT_DIRS[i] for i in sorted(REAGENT_DIRS)], target_length=TARGET_LENGTH
)

MODELS = {
    path.basename(model_path): tf.keras.models.load_model(model_path)
    for model_path in glob.glob(path.join(MODEL_HOME, "*.tf"))
}


def test_point(*spectra: NMRSpectrum):
    return np.vstack([s.spectrum.real for s in spectra])[np.newaxis, ...]


def nmr_process(folder: str, model: tf.keras.Model) -> bool:
    rxn_spec = process_nmr(NMRSpectrum(folder), length=TARGET_LENGTH)
    reagents = [p for p in path.basename(folder).split("_")[1].split("-")]
    sms = reduce(add, [REAGENT_SPECTRA[int(i)] for i in reagents]).normalize()
    prediction = model.predict(test_point(rxn_spec, sms))[0, 0]
    return prediction > 0.5 and 1.0 or 0.0
