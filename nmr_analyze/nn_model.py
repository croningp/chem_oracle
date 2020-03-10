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


def process_nmr(nmr_path, min_length=None):
    r = (
        NMRSpectrum(nmr_path)
        .crop(0, 12, inplace=True)
        .remove_peak(2.5, rel_height=0.9999, inplace=True, cut=False)
        .autophase(inplace=True)
        .cut(1.3, 3.65, inplace=True)  # remove DMSO peak region
        .normalize(inplace=True)
    )
    if min_length is not None and len(r) > min_length:
        r.spectrum = r.spectrum[:min_length]
        r.xscale = r.xscale[:min_length]
        r.length = min_length
    return r


MIN_LENGTH = 3921
DATA_FOLDER = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data"
MODEL_PATH = "/mnt/orkney1/NMRModel/model8.tf"
REAGENT_DIRS = {
    int(path.basename(p).split("_")[1]): p
    for p in glob.glob(path.join(DATA_FOLDER, "reagents", "*_1H"))
}
REAGENT_SPECTRA = [
    process_nmr(REAGENT_DIRS[i], min_length=MIN_LENGTH) for i in sorted(REAGENT_DIRS)
]
MODEL = tf.keras.models.load_model(MODEL_PATH)


def nmr_process(folder: str) -> bool:
    rxn_spec = process_nmr(folder, min_length=MIN_LENGTH)
    reagents = [p for p in path.basename(folder).split("_")[1].split("-")]
    sms = reduce(add, [REAGENT_SPECTRA[int(i)] for i in reagents]).normalize()
    test_point = np.vstack([sms.spectrum.real, rxn_spec.spectrum.real])
    return MODEL.predict(test_point[np.newaxis, ...])[0, 0] > 0.5
