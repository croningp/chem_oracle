import glob
import os
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve

from hplc_analyze import hplc_dario as hplc_processing
from ms_analyze.ms import MassSpectra
from nmr_analyze.nmr_analysis import NMRSpectrum
from nmr_analyze.nn_model import process_nmr, nmr_process, MODELS
import matplotlib.ticker as ticker

DEFAULT_MODEL = MODELS["model19-11-13.tf"]

DATA_HOME = "/home/group/Discovery data/data2"
MAIN_FOLDER = DATA_HOME
REAGENTS_FOLDER = os.path.join(DATA_HOME, "reagents")


name_list = pd.read_excel(os.path.join(MAIN_FOLDER, "data3.xlsx"), engine="openpyxl")

ms_lib = {
    0: (56.0, 42.0, 83.0),
    1: (101.0, 71.0, 42.0, 118.0, 159.0),
    2: (42.0, 83.0, 79.0),
    3: (225.0, 226.0, 42.0, 266.0, 83.0, 56.0, 121.0),
    4: (64.0, 42.0, 79.0, 47.0, 63.0),
    5: (42.0, 143.0, 79.0, 83.0, 56.0),
    6: (103.0, 104.0, 42.0, 45.0, 79.0, 83.0, 56.0, 91.0),
    7: (64.0, 42.0, 47.0, 79.0, 83.0, 56.0),
    8: (
        98.0,
        69.0,
        166.0,
        41.0,
        42.0,
        44.0,
        140.0,
        270.0,
        80.0,
        112.0,
        82.0,
        83.0,
        114.0,
        241.0,
        56.0,
    ),
    9: (199.0, 200.0, 201.0, 42.0, 167.0, 241.0),
    10: (56.0, 42.0, 83.0),
    11: (42.0, 150.0),
    12: (100.0, 70.0, 72.0, 42.0, 79.0, 83.0, 56.0),
    13: (56.0, 137.0, 42.0, 83.0),
    14: (56.0, 42.0, 83.0),
    15: (64.0, 42.0, 79.0, 47.0),
    16: (42.0, 79.0, 83.0, 56.0, 155.0),
    17: (42.0, 139.0, 111.0, 83.0, 118.0, 56.0, 91.0),
    18: (
        64.0,
        312.0,
        42.0,
        141.0,
        142.0,
        79.0,
        143.0,
        145.0,
        147.0,
        83.0,
        310.0,
        56.0,
        314.0,
    ),
    19: (216.0, 215.0),
}


def get_ms(path):
    """
    Returns a MassSpectra object from a path
    """
    try:
        file = glob.glob(os.path.join(path, "*_is1.npz"))[0]
    except IndexError:
        # in case only one polarity (+ or -) is collected
        file = glob.glob(os.path.join(path, "*.npz"))[0]

    spec = MassSpectra.from_npz(file)
    return spec


def name_to_hplc(ms_name):
    """
    Converts MS path name to HPLC
    """
    root = ms_name.split("_")[:-1]
    root.append("HPLC")
    return "_".join(root)


def name_to_nmr(ms_name):
    """
    converts MS path name to NMR
    """
    root = ms_name.split("_")[:-1]
    root.append("1H")
    return "_".join(root)


def get_reagent_file(n):
    """
    Takes the reagent number and returns its MS path
    """
    msreagents_paths = [
        i for i in glob.glob(os.path.join(REAGENTS_FOLDER, "*_MS")) if "BLANK" not in i
    ]
    for i in msreagents_paths:
        if i.split("_")[1] == n or int(i.split("_")[1]) == n:
            return i


def get_reagent_spectrum(n):
    """
    Takes the reagent number return its MassSpectra object
    """
    path = get_reagent_file(n)
    return get_ms(path)


def get_reagent_weight(n):
    """
    Takes the reagents number return its weight
    """
    return name_list[name_list["reagent_number"] == n]["MW"].tolist()[0]


def get_reagent_name(r):
    """
    Takes the reagent number returns its name
    """
    return name_list[name_list["reagent_number"] == r]["reagent_name"].tolist()[0]


def forge_blank_name(filename):
    """
    Converts MS path to the relative blank path
    """
    parts = filename.split("_")
    parts.insert(2, "BLANK")
    return "_".join(parts)


def baseline_als(y, lam, p, niter=10):
    """
    Function to calculate the baseline
    returns anarray with the baseline y-values
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        return z


def extract_peaks(spec, start, end, height=0.1):
    """
    takes a chromatogram and an interval of times, returns the highest ms peaks after integration
    """
    integral = spec.integrate(start, end)
    peaks = integral.intensities / max(integral.intensities)  # y axis of the spectrum
    masses = integral.masses  # x axis of the spectrum
    # position of the peaks and intensity
    found = find_peaks(peaks, height=height)
    peaks_mass = np.take(masses, found[0])  # masses of the peaks
    return peaks_mass, peaks, found, masses


def filter_out_peaks(mix_p, reag_p):
    """
    given mix and reagents peaks, remove the reagents from mix, within a proximity error, after delay
    """
    new_peaks = []
    delay = 0.2
    proximity = 0.3
    for mp in mix_p:
        for rp in reag_p:
            if rp + delay - proximity < mp < rp + delay + proximity:
                break
        else:
            new_peaks.append(mp)
    return new_peaks


def remove_naphtalene_contamination(spec, peaks):
    """
    removes the naphtalene peak if present from a list of TIC peaks
    """
    DT = 1
    napht = 10.75
    filtered = []
    for p in peaks:
        flag = False
        if napht - DT < p < napht + DT:
            peaks_mass, _, _, _ = extract_peaks(spec, p - DT, p + DT, height=0.5)
            if 215 in [int(i) for i in peaks_mass]:
                flag = True
        if not flag:
            filtered.append(p)
    return filtered


def find_TIC_peaks(tic):
    """
    Takes a TIC object, corrects the baseline and return the peaks
    """
    z = baseline_als(tic[1], 100000, 1)  # baseline correction
    TIC_flat = tic[1] - z  # applying the baseline
    TIC_noise = savgol_filter(TIC_flat, 15, 2)
    TIC_p = find_peaks(TIC_noise, height=0.1)  # finding peaks on corrected tic
    return TIC_p


class ProcessMS:
    def __init__(self, s):
        self.s = s
        self.spec = get_ms(self.s)
        self.reagents = [x for x in os.path.basename(self.s).split("_")[1].split("-")]
        self.TIC = self.spec.chromatogram(normalize=True)

    def get_hplc_data(self):
        """
        Assign internal variables to HPLC data, mix, reconstructed, reagents peaks and new peaks
        """

        try:
            (
                self.hplc_mix,
                self.hplc_diff,
                self.hplc_recon,
                _,
            ) = hplc_processing.filter_spectrum(name_to_hplc(self.s))
        except (NameError, FileNotFoundError) as e:
            print("no HPLC found")
            self.has_hplc = False
            return
        self.hplc_new_p = find_peaks(
            self.hplc_diff / max(self.hplc_recon), height=0.1
        )  # rt of NEW peaks from hplc data
        self.hplc_reactivity = False if len(self.hplc_new_p[0]) == 0 else True
        self.hplc_r_p = find_peaks(
            self.hplc_recon / max(self.hplc_recon), height=0.05
        )  # rt of REAGENT peaks from hplc data
        self.has_hplc = True

    def get_nmr_data(self, nmr_react):
        self.nmr_mix = process_nmr(NMRSpectrum(name_to_nmr(self.s)))
        self.nmr_yscale = np.real(self.nmr_mix.spectrum)[:2430]
        self.nmr_xscale = self.nmr_mix.xscale[:2430]
        nmr_reagents = np.array(
            [
                np.real(
                    process_nmr(
                        NMRSpectrum(name_to_nmr(get_reagent_file(reagent)))
                    ).spectrum
                )[:2430]
                for reagent in self.reagents
            ]
        )
        self.nmr_recon = np.sum(nmr_reagents, axis=0)
        if nmr_react:
            self.nmr_reactivity = nmr_process(
                folder=name_to_nmr(self.s), model=DEFAULT_MODEL
            )

    def remove_known_masses(self):
        """
        Removes blank and reagents masses from the spectrum
        """
        blank_masses = self.find_blanks_masses()
        reag_lib_masses = self.find_reagents_lib_masses()
        reag_masses = self.find_reagents_masses()

        all_masses = blank_masses + reag_lib_masses + reag_masses
        all_masses = set(all_masses)
        for p in all_masses:
            # removing them from chromatogram
            self.spec.remove_peak(p, delta_mass=1.0)

    def find_reagents_lib_masses(self):
        """
        Gathers reagents masses from library of single reagents injections
        """
        reag_lib_masses = []
        for r in self.reagents:
            reag_lib_masses.extend(ms_lib[int(r)])
        return reag_lib_masses

    def find_blanks_masses(self):
        """
        Gathers masses from the blank run, if any peak in the TIC is present
        """
        b_spec = get_ms(forge_blank_name(self.s))
        b_TIC = b_spec.chromatogram(normalize=True)
        b_TIC_p = find_TIC_peaks(b_TIC)
        bg_masses = [42.0, 83.0, 79.0]  # signals of MeCN and DMSO
        DT = 0.5
        for p in np.take(
            b_TIC[0], b_TIC_p[0]
        ):  # finding ms top peaks around the TIC peaks
            peak_mass, peaks, found, masses = extract_peaks(b_spec, p - DT, p + DT)
            bg_masses.extend(np.rint(peaks))
        return list(set(bg_masses))

    def find_reagents_masses(self):
        """
        Gathers reagents masses from the mix data, taking the rt from the HPLC data
        """
        if not hasattr(self, "hplc_r_p"):
            self.get_hplc_data()
            if not self.has_hplc:
                return []
        reag_masses = []
        DT = 0.5
        for p in np.take(
            self.hplc_mix, self.hplc_r_p[0]
        ):  # finding ms top peaks around the TIC peaks
            peak_mass, peaks, found, masses = extract_peaks(self.spec, p - DT, p + DT)
            reag_masses.extend(np.rint(peak_mass))
        return list(set(reag_masses))

    def filter_tic_peaks(self):
        """
        Cleaning the peak lists in TIC
        removes peaks close to the reagents rt
        removes peaks at start and end
        removes napthalene peak
        """
        if not hasattr(self, "TIC_p"):
            self.TIC_p = find_TIC_peaks(self.TIC)

        if hasattr(self, "hplc_mix"):
            TIC_p_new = filter_out_peaks(
                np.take(self.TIC[0], self.TIC_p[0]),
                np.take(self.hplc_mix[:, 0], self.hplc_r_p[0]),
            )  # removing peaks coming when reagents come out
        else:
            TIC_p_new = self.TIC_p[0]

        TIC_p_new = [
            i for i in TIC_p_new if 3.6 < i < 23
        ]  # removing stuff with solvent front and column fragments at the end
        # TIC_p_new = remove_naphtalene_contamination(
        #    self.spec, TIC_p_new
        # )  # it persists from previous exp
        return TIC_p_new

    def update_tic(self):
        """
        Refresh the TIC variable after manipulating the MassSpectra object
        """
        self.TIC = self.spec.chromatogram(normalize=True)

    def plot_ms(self, ax):
        """
        Plot the MS of new peaks, if present
        """
        if not hasattr(self, "TIC_p_filt"):
            self.TIC_p_filt = self.filter_tic_peaks()
        self.MS_reactivity = False if len(self.TIC_p_filt) == 0 else True
        ax.set_title(str(self.MS_reactivity))
        DT = 0.2
        for p in self.TIC_p_filt:
            peaks_mass, peaks, found, masses = extract_peaks(
                self.spec, p - DT, p + DT, height=0.3
            )
            ax.plot(masses, peaks, label=str(p))
            for f in found[0]:
                ax.annotate(
                    str(int(masses[f])),
                    xy=(masses[f], peaks[f]),
                    xytext=(masses[f], peaks[f] + 0.01),
                )

    def plot_nmr(self, ax, nmr_react):
        """
        plot NMR
        """
        if not hasattr(self, "nmr_reactivity"):
            self.get_nmr_data(nmr_react)
        multiplier = len(self.reagents)
        if nmr_react:
            ax.set_title(str(self.nmr_reactivity))
        ax.plot(self.nmr_xscale, self.nmr_recon, c="blue")
        ax.plot(self.nmr_xscale, self.nmr_yscale * multiplier, c="red")
        ax.set_ylim([-0.1, 0.5])
        ax.invert_xaxis()

    def plot_hplc_tic(self, ax):
        """
        Plot superimposition of cleaned TIC, HPLC mix and HPLC new peaks
        """
        if not hasattr(self, "hplc_mix"):
            self.get_hplc_data()
        if not hasattr(self, "TIC_p_filt"):
            self.TIC_p_filt = self.filter_tic_peaks()

        z = baseline_als(self.TIC[1], 100000, 1)  # baseline correction
        TIC_flat = self.TIC[1] - z  # applying the baseline
        TIC_noise = savgol_filter(TIC_flat, 15, 2)

        ax.plot(
            self.TIC[0],
            TIC_noise,
            c="blue",
            zorder=1,
            alpha=0.4,
            label="MS",
            linestyle="dashed",
        )
        ax.scatter(
            self.TIC_p_filt,
            [-0.2] * len(self.TIC_p_filt),
            zorder=2,
            c="green",
            alpha=0.4,
        )

        if self.has_hplc:
            ax.set_title(str(self.hplc_reactivity))
            ax.plot(
                self.hplc_mix[:, 0],
                (self.hplc_mix[:, 1] / max(self.hplc_recon)),
                c="orange",
                label="mix",
            )
            ax.plot(
                self.hplc_mix[:, 0],
                (self.hplc_diff / max(self.hplc_recon)),
                c="red",
                alpha=0.5,
                label="new",
            )
            ax.plot(
                self.hplc_mix[:, 0],
                self.hplc_recon / max(self.hplc_recon),
                c="green",
                label="reagents",
            )
            # ax.scatter(
            # np.take(self.hplc_mix[:, 0], self.hplc_r_p[0]),
            # self.hplc_r_p[1]["peak_heights"],
            # zorder=2,
            # c="blue",
            # )

            ax.scatter(
                np.take(self.hplc_mix[:, 0], self.hplc_new_p[0]),
                self.hplc_new_p[1]["peak_heights"],
                zorder=2,
                c="red",
            )
        ax.legend()
        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    def calculate_couples(self):
        """
        Prints the calculated masses of combinations of reagents, highlighting matches in new peaks
        """
        if not hasattr(self, "TIC_p_filt"):
            self.TIC_p_filt = self.filter_tic_peaks()

        DT = 0.2
        peaks_masses = []
        for p in self.TIC_p_filt:
            peaks_mass, peaks, found, masses = extract_peaks(
                self.spec, p - DT, p + DT, height=0.3
            )
            peaks_masses.extend(peaks_mass)

        reagents_w = [get_reagent_weight(int(r)) for r in self.reagents]
        reagents_w_couple = [i for i in combinations_with_replacement(reagents_w, 2)]

        DM = 3
        print("reagents = {}".format([get_reagent_name(int(r)) for r in self.reagents]))
        print("single = {}".format(reagents_w))
        print("couples = {}".format([sum(g) for g in reagents_w_couple]))
        if len(self.reagents) == 3:
            reagents_w_couple.append(reagents_w)
            print("full = [{}]".format(sum(reagents_w)))

        for p in peaks_masses:
            for g in reagents_w_couple:
                if sum(g) - DM < p < sum(g) + DM:
                    print("FOUND A MATCH: {} - sum of {} = {}".format(p, g, sum(g)))
                if sum(g) + 23 - DM < p < sum(g) + 23 + DM:
                    print(
                        "FOUND A MATCH WITH Na: {} - sum of {} + 23 = {}".format(
                            p, g, sum(g) + 23
                        )
                    )
                if sum(g) * 2 - DM < p < sum(g) * 2 + DM:
                    print(
                        "FOUND A MATCH DOUBLE MASS: {} - sum of {}*2 = {}".format(
                            p, g, sum(g) * 2
                        )
                    )

    def plot_all(self, nmr_react=False):
        """
        Whip function to visualise all 3 techniques at once
        """
        self.get_hplc_data()
        self.get_nmr_data(nmr_react)

        # self.remove_known_masses()  # remove blank and reagents peaks
        self.update_tic()

        self.TIC_p = find_TIC_peaks(self.TIC)
        self.TIC_p_filt = self.filter_tic_peaks()
        self.MS_reactivity = False if self.TIC_p_filt == [] else True
        print(self.s)
        self.calculate_couples()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))

        self.plot_ms(ax1)
        self.plot_hplc_tic(ax2)
        self.plot_nmr(ax3, nmr_react)
        plt.show()


def ms_reactivity(filename):
    """
    From a MS path removes blanks and reagents masses from the spectra, then cleans the peaks lists of TIC,
    if anything remains return True
    """
    pm = ProcessMS(filename)
    pm.get_hplc_data()
    pm.remove_known_masses()  # remove blank and reagents peaks
    pm.update_tic()
    pm.TIC_p = find_TIC_peaks(pm.TIC)
    pm.TIC_p_filt = pm.filter_tic_peaks()
    return False if len(pm.TIC_p_filt) == 0 else True


if __name__ == "__main__":
    print(ms_reactivity("/mnt/scapa4/group/Hessam Mehr/Data/Discovery/data/0_17-7_MS"))
