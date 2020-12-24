import logging
import struct
from os.path import join
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, fmin


def DEFAULT_XFORM(s):
    return s.crop(0, 12).normalize()


def lorentzian(p, p0, ampl, w):
    x = (p0 - p) / (w / 2)
    return ampl / (1 + x * x)


class NMRSpectrum:
    def __init__(
        self,
        path=None,
        xscale: np.ndarray = None,
        spectrum: np.ndarray = None,
        logger=None,
    ):
        # self.nmr_dir = nmr_dir
        self.nmr_folder = path
        self.spectrum = spectrum
        self.xscale = xscale
        self.logger = logger or logging.getLogger("nmr-logger")

        if path:
            # read spectrum parameters
            spectrum_parameters = open(join(path, "acqu.par"), "r")
            parameters = spectrum_parameters.readlines()
            self.param_dict = {}
            for param in parameters:
                self.param_dict[param.split("= ")[0].strip(" ")] = param.split("= ")[
                    1
                ].strip("\n")

            self.logger.debug(f"Spectrum parameters: {self.param_dict}.")

            # open file with nmr data
            spectrum_path = join(path, "data.1d")
            # open binary file with spectrum
            nmr_data = open(spectrum_path, mode="rb")
            # read first eight bytes
            spectrum = []
            # unpack the data
            while True:
                data = nmr_data.read(4)
                if not data:
                    break
                spectrum.append(struct.unpack("<f", data)[0])
            # remove first eight points and divide data into three parts
            length = len(spectrum[8:]) // 3
            # first column is re(fid); second im(fid)
            fid = np.array(spectrum[length + 8 :]).reshape(-1, 2)
            self.fid = 1j * fid[:, 1] + fid[:, 0]
            # reverse shifted fft to agree with increasing frequency axis
            self.spectrum = np.fft.fftshift(np.fft.fft(self.fid))[::-1]
            self.xscale = self.chemical_shift_scale()
            self.length = len(self.xscale)

    def phase(self, ph0=0.0, ph1=0.0, inplace=False):
        s = self if inplace else self.copy()
        ph = np.linspace(0, 1, len(s)) * ph1 + ph0
        s.spectrum *= np.exp(1j * np.pi * ph)
        return s

    def show(self, imaginary=False, axes=None, **plt_params):
        if not axes:
            fig, axes = plt.subplots()
        else:
            fig = axes.get_figure()
        if imaginary:
            axes.plot(self.xscale, self.spectrum.imag, **plt_params)
        else:
            axes.plot(self.xscale, self.spectrum.real, **plt_params)
        axes.set_xlim(self.xscale.max(), self.xscale.min())
        return fig

    def integrate(self, low_ppm, high_ppm):
        points = (self.xscale > low_ppm) & (self.xscale < high_ppm)
        return np.trapz(self.spectrum.real[points], self.xscale[points])

    def chemical_shift_scale(self) -> np.ndarray:
        b1_freq = float(self.param_dict["b1Freq"])
        dwell_time_us = float(self.param_dict["dwellTime"])
        lowest_frequency_hz = float(self.param_dict["lowestFrequency"])

        lowest_frequency_ppm = lowest_frequency_hz / b1_freq
        sw_hz = 1e6 / dwell_time_us
        sw_ppm = sw_hz / b1_freq
        highest_frequency_ppm = lowest_frequency_ppm + sw_ppm
        return np.linspace(
            start=lowest_frequency_ppm,
            stop=highest_frequency_ppm,
            num=len(self),
        )

    def crop(self, low_ppm: float, high_ppm: float, inplace=False):
        s = self if inplace else self.copy()
        selector = (s.xscale > low_ppm) & (s.xscale < high_ppm)
        s.xscale = s.xscale[selector]
        s.spectrum = s.spectrum[selector]
        return s

    def cut(self, low_ppm: float, high_ppm: float, inplace=False):
        s = self if inplace else self.copy()
        selector = (s.xscale < low_ppm) | (s.xscale > high_ppm)
        s.xscale = s.xscale[selector]
        s.spectrum = s.spectrum[selector]
        return s

    def erase(self, low_ppm: float, high_ppm: float, inplace=False):
        s = self if inplace else self.copy()
        selector = (s.xscale > low_ppm) & (s.xscale < high_ppm)
        p1, p2 = s.spectrum[selector][0], s.spectrum[selector][-1]
        s.spectrum[selector] = p1 + np.linspace(0, p2 - p1, sum(selector))
        return s

    def autophase(self, inplace=False):
        def cost(ph):
            phased_spectrum = self.phase(ph[0], ph[1], inplace=False).spectrum.real
            # penalty for negative areas
            penalty = 1e-14 * (phased_spectrum[phased_spectrum < 0] ** 2).sum()
            first_derivative = np.abs(np.gradient(phased_spectrum))
            prob = first_derivative / first_derivative.sum()
            # penalty for fluctuations
            entropy = np.sum(-prob * np.log(prob))

            self.logger.debug(f"entropy: {entropy}, penalty: {penalty}")
            return entropy + penalty

        new_phase = fmin(cost, [0, 0], disp=False)
        return self.phase(new_phase[0], new_phase[1], inplace=inplace)

    def remove_peak(
        self,
        peak_ppm: float,
        window_ppm: float = 0.1,
        cut: bool = True,
        inplace=False,
        **find_peaks_params,
    ):
        s = self if inplace else self.copy()
        window = (peak_ppm - window_ppm, peak_ppm + window_ppm)
        params = {
            "height": 0.01 * s.spectrum.real.max(),
            "width": 5,
            "rel_height": 0.995,
        }
        params.update(find_peaks_params)
        peaks, data = self.find_peaks(window=window, **params)

        peak_idx = int(peaks[0])
        peak_width = int(data["widths"][0]) * 2
        lo, hi = (peak_idx - peak_width), (peak_idx + peak_width)
        if cut:
            s.cut(s.xscale[lo], s.xscale[hi], inplace=True)
        else:
            vals = np.linspace(s[lo], s[hi], hi - lo)
            s[lo:hi] = vals
        return s

    def find_peaks(self, window: Tuple[float, float] = None, **find_peaks_params):
        signal = self.spectrum.real.copy()
        if window:
            m, M = min(*window), max(*window)
            signal[self.xscale < m] = 0
            signal[self.xscale > M] = 0
        return scipy.signal.find_peaks(signal, **find_peaks_params)

    def fit_lorentzian(self, x, y):
        initial = [x, y, 0.07]
        params, pcov = curve_fit(
            lorentzian, self.xscale, self.spectrum, initial, maxfev=5000
        )
        return params

    def normalize(self, inplace=False):
        s = self if inplace else self.copy()
        s.spectrum /= self.spectrum.real.max()
        return s

    def resize(self, length, inplace=False):
        s = self if inplace else self.copy()
        interpolator = interp1d(s.xscale, s.spectrum)
        s.xscale = np.linspace(s.xscale[0], s.xscale[-1], length)
        s.spectrum = interpolator(s.xscale)
        s.length = length
        return s

    def shift(
        self,
        delta: int,
        inplace=False,
        circular=False,
        fill_value: Union[float, np.ndarray] = 0.0,
    ) -> NMRSpectrum:
        s = self if inplace else self.copy()
        if delta == 0:
            return self
        if circular:
            np.roll(s.spectrum, delta)
        elif delta > 0:
            s.spectrum[delta:] = s.spectrum[:-delta]
            s.spectrum[:delta] = fill_value
        else:
            s.spectrum[:delta] = s.spectrum[-delta:]
            s.spectrum[delta:] = fill_value
        return s

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return NMRSpectrum(
            xscale=self.xscale.copy(), spectrum=self.spectrum.copy(), logger=self.logger
        )

    def __sub__(self, other: NMRSpectrum) -> NMRSpectrum:
        s = self.copy()
        s.spectrum -= other.spectrum
        return s

    def __add__(self, other: NMRSpectrum) -> NMRSpectrum:
        s = self.copy()
        s.spectrum += other.spectrum
        return s

    def __mul__(self, other: Union[NMRSpectrum, float]) -> NMRSpectrum:
        s = self.copy()
        if isinstance(other, NMRSpectrum):
            s.spectrum *= other.spectrum
            return s
        else:
            s.spectrum *= other
            return s

    def __rmul__(self, other: Union[NMRSpectrum, float]) -> NMRSpectrum:
        return self.__mul__(other)

    def __len__(self):
        return len(self.spectrum)

    def __getitem__(self, item):
        return self.spectrum[item]

    def __setitem__(self, item, value):
        self.spectrum[item] = value


class NMRDataset:
    def __init__(
        self,
        dirs,
        adjust_length=True,
        dtype="float32",
        target_length=None,
        transform: Callable[[NMRSpectrum], NMRSpectrum] = DEFAULT_XFORM,
    ):
        self.dirs = dirs
        self.spectra = [transform(NMRSpectrum(d)) for d in self.dirs]
        if adjust_length:
            self.min_length = target_length or min(len(s) for s in self.spectra)
            self.spectra = [
                s.resize(self.min_length, inplace=True) for s in self.spectra
            ]
        self.matrix = np.vstack(
            [s.spectrum.real[: self.min_length] for s in self.spectra]
        ).astype(dtype, casting="same_kind")

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
