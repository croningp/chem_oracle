import logging
import numpy as np

from sklearn.decomposition import NMF
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


class MassSpectra:
    """
    Represents a series of mass spectra, typically from an LC/GC-MS experiment.
    """

    def __init__(
        self,
        retention_times: np.ndarray,
        masses: np.ndarray,
        spectra: np.ndarray,
        scale=None,
    ):
        """Class representing a series of mass spectra in an HPLC/MS experiment.
        Each mass spectrum corresponds to a specific retention time.

        Args:
            retention_times (np.ndarray): Retention times in minutes.
            masses (np.ndarray): Masses for which intensities are measured.
            spectra (np.ndarray): Intensities for each mass for each spectrum.
                spectra[i, j] is the intensity of the j-th ion in the i-th spectrum.
            scale ([type], optional): [description]. TODO
        """
        self.retention_times = retention_times
        self.masses = masses
        self.spectra = spectra
        # highest peak in spectrum
        self.scale = scale or spectra.max()

    @classmethod
    def from_advion(cls, data):
        retention_times = data.retention_times() / 60.0
        masses = data.masses()
        spectra = data.spectra()
        return cls(retention_times, masses, spectra)

    @classmethod
    def from_npz(cls, filename: str):
        """
        Create a MassSpectra instance from a numpy npz file.
        """
        data = np.load(filename)
        retention_times = data["retention_times"] / 60.0
        masses = data["masses"]
        spectra = data["spectra"]
        return cls(retention_times, masses, spectra)

    def discretize(self, delta_mass: float):
        """
        Bin the mass axis with resolution 

        Parameters
        ----------
        delta_mass (float): The desired mass resolution.

        Returns
        -------
        A new instance of MassSpectra with the new mass resolution and the same
        number of mass spectra.
        """
        new_masses = np.arange(self.masses.min(), self.masses.max(), delta_mass)
        new_spectra = np.zeros((len(self.retention_times), len(new_masses)))
        for i, m in enumerate(new_masses):
            new_spectra[:, i] = self.spectra[
                :, (self.masses > m) & (self.masses < m + delta_mass)
            ].sum(axis=1)
        return MassSpectra(self.retention_times, new_masses, new_spectra)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return MassSpectrum(self.masses, self.spectra[idx, :])
        else:
            return [MassSpectrum(self.masses, self.spectra[i, :]) for i in idx]

    @property
    def time_resolution(self):
        return self.retention_times[1] - self.retention_times[0]

    def get_slice(self, t_start: float, t_end: float):
        """
        Return the portion of the spectra with retention time between t_start
        and t_end.

        Returns
        -------
        A new instance of MassSpectra containing the time slice t_start...t_end.
        """
        selector = (t_start < self.retention_times) & (self.retention_times < t_end)
        return MassSpectra(
            self.retention_times[selector],
            self.masses,
            self.spectra[selector, :],
            self.scale,
        )

    def chromatogram(self, normalize: bool = False):
        """
        Returns a 2 x N matrix whose first row represents the retention time
        (in minutes) and whose second row represents the total ion count for
        the corresponding retention time. The ion count is normalized so the
        highest count is 1.0 if `normalize` is set to `True`.

        Parameters
        ----------
        normalize (bool): 
        """
        TICs = self.spectra.sum(axis=1)
        return np.stack(
            [self.retention_times, TICs / TICs.max() if normalize else TICs]
        )

    def remove_peak(self, mass: float, delta_mass: float = 2.5):
        """
        Remove all peaks whose mass lies between mass - delta_mass and
        mass + delta_mass from all spectra.
        """
        indices = (self.masses > mass - delta_mass) & (self.masses < mass + delta_mass)
        self.spectra[:, indices] = 0.0
        return self

    def integrate(self, t_start: float, t_end: float, normalize=False):
        intensities = self.get_slice(t_start, t_end).spectra.mean(axis=0)
        return MassSpectrum(
            self.masses, intensities / intensities.max() if normalize else intensities
        )

    def remove_background(self, bg_t1: float = 0.0, bg_t2: float = 1.0, Δ: float = 2.5):
        bg = self.integrate(bg_t1, bg_t2, normalize=False)
        bg.intensities /= self.scale
        peaks = bg.find_peaks(height=0.1, distance=3)
        for peak in peaks:
            logging.debug(f"Removing peak at {self.masses[peak]}")
            self.remove_peak(self.masses[peak], delta_mass=Δ)
        return self

    def TIC_peaks(self, height=0.2, peak_Δt=0.5, prominence=0.15, **options):
        times, chromatogram = self.chromatogram(normalize=False)
        chromatogram /= self.scale
        return find_peaks(
            chromatogram,
            height=height,
            distance=int(peak_Δt / self.time_resolution),
            prominence=prominence,
            **options,
        )[0]

    def main_masses(self):
        max_intensities = self.spectra.max(axis=0)
        max_intensities /= max_intensities.max()
        peaks = find_peaks(max_intensities, distance=10, height=0.1)[0]
        return peaks, self.masses[peaks]

    def find_components(self, num_components: int):
        """
        Uses non-negative matrix factorization (NMF) to decompose the given series of
        mass spectra into independent groups of peaks.
        
        Returns:
        Ion chromatogram for the spectra in each group and the corresponding
        mass spectra.
        """
        model = NMF(num_components)
        transformed = model.fit_transform(self.spectra.T)
        reconstruction = model.inverse_transform(transformed).T.sum(axis=1)
        ref = self.chromatogram()[1]
        error = np.abs(reconstruction - ref).sum() / ref.sum()
        return (
            model.components_.T,
            [
                MassSpectrum(self.masses, transformed[:, i])
                for i in range(num_components)
            ],
            error,
        )

    def find_components_adaptive(self, max_error, min_components=3, max_components=15):
        for n in range(min_components, max_components + 1):
            components, spectra, error = self.find_components(n)
            if error < max_error:
                return components, spectra, error
        raise Exception(f"Could not reach error < {max_error}.")


class MassSpectrum:
    def __init__(self, masses: np.array, intensities: np.array):
        self.masses = masses
        self.intensities = intensities

    def plot(self, normalized=False):
        intesities = self.intensities
        if normalized:
            plt.plot(self.masses, intensities / self.intensities.max())
        else:
            plt.plot(self.masses, self.intensities)

    def find_peaks(self, **options):
        return find_peaks(self.intensities / self.intensities.max(), **options)[0]

    def find_main_peak(self):
        peak_idx = np.argmax(self.intensities)
        return peak_idx

    def __getitem__(self, idx):
        return self.intensities[idx]
