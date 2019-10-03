import numpy as np
from scipy import signal
import math
import pandas as pd
from matplotlib import pyplot as plt
from os import path

import nmr_analysis as na

spec_lenght = 4878

def get_nmr(file_path):
    spectrum = na.nmr_spectrum(file_path)
    na.default_processing(spectrum, solvent="DMSO")
    y = spectrum.spectrum
    x = spectrum.X_scale
    return x, y


def get_theoretical_nmr(reagents, reagent_folder):
    theoretical = np.zeros(spec_lenght)
    reagents_filt = []
    volume = len(reagents)
    for reagent in reagents:
        _, reag_nmr = get_nmr(path.join(reagent_folder, reagent)) #we are interested in the y data, multiplied by the volume used
        theoretical = theoretical + (reag_nmr)
        reagents_filt.append(reagent)
    theoretical_norm = theoretical / volume
    theoretical_norm[0:1550] = np.zeros(1550)
    return theoretical_norm

def make_input_matrix(file_path, reagents, reagent_folder):
    nmr_datax, nmr_datay = get_nmr(file_path)
    if len(nmr_datay) < spec_lenght:
        nmr_y = nmr_datay
        for i in range(spec_lenght - len(nmr_datay)):
            nmr_y.append(nmr_datay[0])
        nmr_datay = np.array(nmr_y)
    theoretical = get_theoretical_nmr(reagents, reagent_folder)
    if len(theoretical) < spec_lenght:
        theo_list = theoretical.tolist()
        for i in range(spec_lenght - len(theoretical)):
            theo_list.append(theo_list[0])
        theoretical = np.array(theo_list)

    nmr_datay = signal.resample_poly(
        nmr_datay, 1000, 18040)
    theoretical = signal.resample_poly(
        theoretical, 1000, 18040)

    # normalize to 1.0
    avg_max = max(max(theoretical), max(nmr_datay))
    theoretical = theoretical / avg_max
    nmr_datay = nmr_datay / avg_max

    # Reshape amd concatenate
    nmr_datay = nmr_datay.reshape(1, -1)
    theoretical = theoretical.reshape(1, -1)
    concatented = np.concatenate(
        (theoretical, nmr_datay), axis=0)
    concatenated = concatented.reshape(2, len(theoretical[0]))
    data_x = np.array([concatenated])

    return data_x


class DataManager:
    """ Data manager to get batches for training"""

    def __init__(self, data_x, data_y, random_shift=False):
        self.data_x = data_x
        self.data_y = data_y
        self.size = len(self.data_x)
        self.curr_idx = 0
        self.random_shift = random_shift
        self.spec_length = data_x[-1]

    def next_batch(self, batch_size):
        """ Generates next batch of data.
        Args:
            batch_size: size of the batch
        Returns:
            New batch of data 
        """

        batch_x = self.data_x[self.curr_idx : self.curr_idx + batch_size]
        batch_y = self.data_y[self.curr_idx : self.curr_idx + batch_size]

        if self.random_shift:
            random_idx = np.random.randint(low=0, high=271)
            batch_x = (batch_x[:, :, -random_idx:], batch_x[:, :, :-random_idx])
            batch_x = np.concatenate(batch_x, axis=2)

        self.curr_idx = (self.curr_idx + batch_size) % self.size
        return batch_x, batch_y

