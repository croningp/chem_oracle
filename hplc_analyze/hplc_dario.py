import glob
import logging
import os
from os import path

import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks

from hplc_analyze import chemstation

full_lib_1 = {
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

full_lib_2 = {
    0: [(18.7, 18.1)],
    1: [(12.0, 11.4)],
    2: [(3.0, 2.0)],
    3: [(13.6, 12.7), (5.5, 4.2), (10.2, 9.8)],
    4: [(4.2, 2.5)],
    5: [(15.2, 14.0), (10.8, 10.0), (8.5, 7.7), (5.0, 2.5)],
    6: [(5.4, 4.0)],
    7: [(17.3, 17.0), (16.2, 15.5), (14.4, 14.0)],
    8: [(16.0, 15.5), (3.2, 2.7)],
    9: [(14.5, 14.1), (12.4, 11.0), (3.7, 3.0)],
    10: [(5.4, 3.9)],
    11: [(17.2, 16.2)],
    12: [(21.7, 21.3)],
    13: [(18.7, 18.4)],
    14: [(3.0, 2.0)],
    15: [(11.8, 10.45)],
    16: [(10.3, 9.4), (7.8, 6.0), (5.5, 3.2)],
    17: [(3.5, 2.5)],
    18: [(3.5, 2.5)],
    19: [(17.9, 17.4), (3.6, 2.4)],
    20: [(15.3, 14.2)],
    21: [(15.0, 14.3)],
    22: [(4.0, 3.0), (8.5, 7.5)],
    23: [(3.5, 2.5)],
}

full_lib_3 = {
    0: [(8.5, 6.8), (13.7, 13.45), (14.5, 14.2), (18.25, 18.0), (18.75, 18.5)],
    1: [(18.3, 18.0)],
    2: [(18.6, 18.0), (19.5, 19.25), (19.9, 19.75)],
    3: [(18.3, 18.0)],
    4: [(14.6, 14.15), (13.0, 12.4)],
    5: [(14.5, 14.25), (12.25, 11.9)],
    6: [(3.0, 2.7)],
    7: [(16.0, 12.4)],
    8: [(13.3, 12.9)],
    9: [(3.0, 2.0)],
    10: [(3.0, 2.0)],
    11: [(3.0, 2.0)],
}

REAGENTS_FOLDER_1 = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/"
reagents_files_1 = {
    i.split("_")[1]: i
    for i in glob.glob(os.path.join(REAGENTS_FOLDER_1, "reagents", "*_HPLC"))
    if "BLANK" not in i
}

REAGENTS_FOLDER_2 = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data2/"
reagents_files_2 = {
    i.split("_")[1]: i
    for i in glob.glob(os.path.join(REAGENTS_FOLDER_2, "reagents", "*_HPLC"))
    if "BLANK" not in i
}

REAGENTS_FOLDER_3 = "/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data3/"
reagents_files_3 = {
    i.split("_")[1]: i
    for i in glob.glob(os.path.join(REAGENTS_FOLDER_3, "reagents", "*_HPLC"))
    if "BLANK" not in i
}

space_manager = {
    "data": {"reagents": reagents_files_1, "lib": full_lib_1},
    "data2": {"reagents": reagents_files_2, "lib": full_lib_2},
    "data3": {"reagents": reagents_files_3, "lib": full_lib_3},
}


def load_hplc(path):
    file = np.load(path + "/DAD1A.npz", allow_pickle=True)
    return file["times"], file["values"]


def get_reagent_file(n, spaceN):
    n = str(n)
    reagents = space_manager[spaceN]["reagents"]
    return reagents[n]


def get_reagents(file):
    reagents = path.basename(file).split("_")[1].split("-")  #
    return reagents


def get_reagent_chromatogram(reagent_name: str, data_n):
    file = space_manager[data_n]["reagents"][reagent_name]
    x, y = load_hplc(file)
    spectrum = np.array([x, y]).T
    return spectrum


def get_spectrum(experiment_dir: str, channel: str = "A"):
    filename = path.join(experiment_dir, "DAD1{}".format(channel))
    npz_file = filename + ".npz"
    if path.exists(npz_file):
        data = np.load(npz_file, allow_pickle=True)
        return np.array([data["times"], data["values"]]).T
    else:
        logging.debug("Not found {}".format(npz_file))
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
    reagents_spec = [get_reagent_chromatogram(name) for name in reagents_names]
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


def hplc_process(file):
    original, diff, recon, filt = filter_spectrum(file)
    new_peaks = find_peaks(diff / max(recon), height=0.4)
    return 0.0 if len(new_peaks[0]) == 0 or len(new_peaks) > 15 else 1.0


def filter_and_plot(file):
    original, diff, recon, filt = filter_spectrum(file)
    fig = plt.figure(figsize=(20, 5))
    plt.plot(original[:, 1] * 1.2, c="blue")
    plt.plot([-50 if i == 0 else 1 for i in filt], c="green")
    plt.plot(diff, c="red")
    # plt.plot(recon, c='green')
    plt.title(file)
    plt.show()


if __name__ == "__main__":
    print(hplc_process("/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/0_17-7_HPLC"))
