import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize
from matplotlib import pylab as plt
import random
import time
from scipy.signal import find_peaks_cwt
import os
from sklearn.preprocessing import LabelEncoder
import glob
from os import path
from scipy.signal import find_peaks

full_lib = {
    0: [(13.6, 12.0)],
    1: [(3.0, 2.0)],
    2: [(4.0, 2.7)],
    3: [(20.1, 19.5), (19.5, 18.8)],
    4: [(4.5, 3.2), (6.7, 5.5)],
    5: [(3.0, 2.0)],
    6: [(14.7, 13.7)],
    7: [(20.5, 19.5), (4.0, 3.3)],
    8: [(13.1, 11.8)],
    9: [(20.5, 19.7)],
    10: [(13.9, 13.1), (17.5, 16.5)],
    11: [(3.5, 2.0)],
    12: [(20.3, 19.7), (4.0, 1.8)],
    13: [(12.3, 11.7), (16.0, 15.0)],
    14: [(19.7, 18.9)],
    15: [(4.5, 2.9)],
    16: [(3.0, 2.0)],
    17: [(16.0, 14.8)],
    18: [(8.2, 7.5), (7.5, 6.5), (24.2, 23.2)],
    19: [(3.5, 2.0)],
}

folder = "/mnt/scapa4/group/Dario Caramelli/Projects/FinderX/data/20180418-1809-photochemical_space/"
# reag_fold = 'Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\002_photo_space\\reagents\\'
MAIN_FOLDER = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/"
REAGENTS_FOLDER = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/reagents"

reactions = [
    i for i in glob.glob(os.path.join(MAIN_FOLDER, "*_HPLC")) if "BLANK" not in i
]
reactions_blanks = glob.glob(os.path.join(MAIN_FOLDER, "*BLANK_HPLC"))
reagents = [
    i for i in glob.glob(os.path.join(REAGENTS_FOLDER, "*_HPLC")) if "BLANK" not in i
]
reagents_blanks = glob.glob(os.path.join(REAGENTS_FOLDER, "*BLANK_HPLC"))


def load_hplc(path):
    file = np.load(path + "/DAD1A.npz")
    return file["times"], file["values"]


def get_reagent_file(n):
    for i in reagents:
        if i.split("_")[1] == n or int(i.split("_")[1]) == n:
            return i


def get_reagents(file):
    reagents = [
        x
        for x in path.basename(file).split("_")[1].split("-")
        if x not in ["1", "5", "16"]
    ]  #
    return reagents


def get_reagent_spectrum(n):
    file = get_reagent_file(n)
    x, y = load_hplc(file)
    spectrum = np.array([x, y]).T
    return spectrum


def get_spectrum(experiment_dir: str, channel: str = "A"):
    filename = path.join(experiment_dir, f"DAD1{channel}")
    npz_file = filename + ".npz"
    if path.exists(npz_file):
        data = np.load(npz_file)
        return np.array([data["times"], data["values"]]).T
    else:
        print(f"Not found {npz_file}")
        ch_file = filename + ".ch"
        data = chemstation.CHFile(ch_file)
        np.savez_compressed(npz_file, times=data.times, values=data.values)
        return np.array([data.times, data.values]).T


def forge_blank_name(filename):
    parts = filename.split("_")
    parts.insert(2, "BLANK")
    return "_".join(parts)


def region(spectrum, cs_range):
    indices = (spectrum[:, 0] < cs_range[0]) & (spectrum[:, 0] > cs_range[1])
    start = np.where(indices)[0][0]
    end = np.where(indices)[0][-1]
    data = spectrum[indices, :][:, 1]
    return data, start, end


def makelib(
    spectra, regions
):  # makes a list of regions [[[points of region],start,end],[points of region],start,end]], [...]] for all reagents
    return [[region(s, r) for r in regions[i]] for (i, s) in enumerate(spectra)]


def get_exp_stuff(file):
    spec = get_spectrum(file)
    if min(spec[:, 1]) > 0:
        spec[:, 1] = spec[:, 1] - min(spec[:, 1])  # correct the baseline
    reagents_names = get_reagents(file)
    if reagents_names == []:
        return False, spec
    reagents_spec = [get_reagent_spectrum(name) for name in reagents_names]
    lib = makelib(reagents_spec, [full_lib[int(name)] for name in reagents_names])
    return lib, spec


def adjusted(region, shift):
    reg_st = region[1] + int(shift * 100)
    reg_end = region[2] + int(shift * 100)
    reg_scaled = region[0]
    return [reg_scaled, reg_st, reg_end]


def synthesize(reg, shifts, length):
    result = np.zeros(length)
    for i, r in enumerate(reg):
        a = adjusted(r, shifts[i])
        try:
            before = np.zeros(a[1])
        except ValueError:  # this is to handle peaks shifting before the spectrums start
            before = np.zeros(0)
            a[0] = a[0][abs(a[1]) :]  # cutting out the negative part
        try:
            after = np.zeros(length - (len(before) + len(a[0])))
        except ValueError:  # this is to handle peaks shifting after the spectrum length
            a[0] = a[0][: length - (len(before) + len(a[0]))]
            after = np.zeros(0)
        merged = np.concatenate([before, a[0], after], axis=0)
        if merged.shape[0] > length:
            merged = np.array(
                [3000 for i in range(length)]
            )  # setting it crazy high so optimizer goes away
        result = np.sum(np.stack([result, merged], axis=1), axis=1)
    return result


def get_loss(shifts):
    flag = False
    for j in range(len(shifts)):
        for i in shifts[j]:
            if i > 1 or i < -1:
                flag = True
    if flag:
        return 10
    else:
        return 0


def get_error(x, *params):  # x=list of parameters
    x_ = list(x)

    (
        spec,
        lib,
    ) = params  # spec = reaction mix, lib = list of region data, in list for reagents
    # [  [  [data,start,end],...  ],  [  [data,start,end],...]  ]
    l = len(lib)  # number of reagents
    N = len(spec)  # spectrum lenght

    shifts = [[x_.pop(0) for j in lib[i]] for i in range(l)]
    guess = np.sum(
        np.stack([synthesize(r, shifts[i], N) for i, r in enumerate(lib)], axis=1),
        axis=1,
    )
    err = np.sqrt(np.sum((spec[:] - guess) ** 2) / N)
    return err + get_loss(shifts)


def make_annealing(product, mylib):
    params = (product, mylib)
    l = len(mylib)  # number of reagents
    N = len(product[:1])  # spectrum lenght
    shifts = [[0.05 for j in range(len(mylib[i]))] for i in range(l)]
    x0 = [y for x in shifts for y in x]
    res = minimize(get_error, x0, args=params, method="Powell")
    if res.x.shape == ():
        return np.array([res.x])
    else:
        return res.x


def from_x_to_spec(x, lib, spec):
    l = len(lib)
    N = len(spec)
    x_ = list(x)
    shifts = [[x_.pop(0) for j in lib[i]] for i in range(l)]
    guess = np.sum(
        np.stack([synthesize(r, shifts[i], N) for i, r in enumerate(lib)], axis=1),
        axis=1,
    )
    return guess


def from_recon_to_filter(recon):  # makes 0/1 mask from reconstructred spectrum
    return np.array([1 if i == 0 else 0 for i in recon])


def apply_filter(spec, filt):
    result = spec * filt
    return result


def filter_spectrum(file):
    """
    Removes the reagent peaks from the HPLC spectrum
    returns:
    spec: original HPLC spectrum
    diff: spectrum without reagents peaks
    recon: reconstructed spectrum made with sum of reagents, shifted to adapt to mixture
    filt: 0/1 mask made from recon, to apply on spec to get diff
    """
    lib, spec = get_exp_stuff(file)
    if not lib:
        print(file + ": Reagents dont have spectra")
        return spec, spec[:, 1], spec[:, 1], np.zeros(len(spec[:, 1]))
    res = make_annealing(spec[:, 1], lib)
    recon = from_x_to_spec(res, lib, spec)
    filt = from_recon_to_filter(recon)
    diff = apply_filter(spec[:, 1], filt)
    return spec, diff, recon, filt


def hplc_reactivity(file):
    original, diff, recon, filt = filter_spectrum(file)
    new_peaks = find_peaks(diff / max(original[:, 1]), height=0.1)
    return False if len(new_peaks) == 0 or len(new_peaks) > 100 else True


def filter_and_plot(file):
    original, diff, recon, filt = filter_spectrum(file)
    fig = plt.figure(figsize=(20, 5))
    plt.plot(original[:, 1] * 1.2, c="blue")
    plt.plot([-50 if i == 0 else 1 for i in filt], c="green")
    plt.plot(diff, c="red")
    # plt.plot(recon, c='green')
    plt.title(file)
    plt.show()
