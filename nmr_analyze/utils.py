import numpy as np
from scipy import signal
import math
import pandas as pd
from matplotlib import pyplot as plt

import nmr_analyze.nmr_analysis as na

spec_lenght = 4878

def get_nmr(file_path):
    spectrum = na.nmr_spectrum(file_path)
    na.default_processing(spectrum, solvent='DMSO')
    y = spectrum.spectrum
    x = spectrum.X_scale
    return x,y

def get_theoretical_nmr(reagents):
    reagent_fold = 'Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\002_photo_space\\reagents\\'
    theoretical = np.zeros(spec_lenght)
    reagents_filt = []
    volume = len(reagents)*2
    for reagent in reagents:
        reag_nmr = np.loadtxt(reagent_fold+reagent+'-NMR-0.csv', delimiter=',')*2 #multiplying by the volume of each reagent
        theoretical = theoretical + reag_nmr[:,1]
        reagents_filt.append(reagent)
    theoretical_norm = theoretical/volume
    theoretical_norm[0:1550] = np.zeros(1550)
    return theoretical_norm

def raw_nmr_to_dataframe(file_path, reagents):
    nmr_datax, nmr_datay = get_nmr(file_path)
    if len(nmr_datay) < 4878:
        nmr_y = nmr_datay
        for i in range(4878 - len(nmr_datay)):
            nmr_y.append(nmr_datay[0])
        nmr_datay = np.array(nmr_y)
    theoretical = get_theoretical_nmr(reagents)
    if len(theoretical) < 4878:
        theo_list = theoretical.tolist()
        for i in range(4878 - len(theoretical)):
            theo_list.append(theo_list[0])
        theoretical = np.array(theo_list)
    df = pd.DataFrame(columns=['experimental_spectrum', 'theoretical_spectrum', 'reactivity_label'])
    df.loc[len(df)] = [str(list(nmr_datay)),str(list(theoretical)), '[0,0,0,0]']
    return df

def read_data(dataframe):
    """" Reads and process the nmr data. Performs spectra dimensionality reduction
    and normalizes them to 1.0.
    
        Args:
            dataframe: a pandas dataframe with theoretical and experimental data
                       of reaction mixtures
        Returns:
            data_x: stacked theoretical and experimental spectra of shape [num_examples, 2, 271]
            data_y: one-hot encoded target reactity of shape [num_examples, 4]

    """
    data_x = []
    data_y = []

    for line in range(len(dataframe)):
        # Read the spectra
        experimental_spectrum = np.array(eval(dataframe.loc[line].experimental_spectrum))
        theoretical_spectrum = np.array(eval(dataframe.loc[line].theoretical_spectrum))

        # Downscale the spectra to 271
        if len(experimental_spectrum) == 4878:
            experimental_spectrum = signal.resample_poly(experimental_spectrum, 1000, 18040)
            theoretical_spectrum = signal.resample_poly(theoretical_spectrum, 1000, 18040)
        else:
            experimental_spectrum = signal.resample_poly(experimental_spectrum, 1, 10)
            theoretical_spectrum = signal.resample_poly(theoretical_spectrum, 1, 10)
    
        # normalize to 1.0
        avg_max =  max(max(theoretical_spectrum) , max(experimental_spectrum))
        theoretical_spectrum = theoretical_spectrum / avg_max
        experimental_spectrum = experimental_spectrum / avg_max
        
        # Reshape amd concatenate 
        experimental_spectrum = experimental_spectrum.reshape(1, -1)
        theoretical_spectrum = theoretical_spectrum.reshape(1, -1)
        concatented = np.concatenate((theoretical_spectrum, experimental_spectrum), axis=0)
        concatenated = concatented.reshape(2, len(theoretical_spectrum[0]))
        data_x.append(concatenated)
        
        y = np.array(eval(dataframe.loc[line].reactivity_label))
        data_y.append(y)

    # Convert to numpy arrays
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    return data_x, data_y

def training_validation_test_split(data_x, data_y, training_size,
                                   validation_size, seed=125):
    
    """ Performs random split into training, validation, amd test set.
        
        Args:
            data_x: an array with X data
            data_y: an array with targets Y
            training_size: how many percent of the data should go to traing set
            validation_size: how many percent of the data should go to validation set
            seed: random generator seed for reproducibilty
        Return:
             An array of data, targets split into three sets.
             
    """
    # Make sure that traing + validation is less than 1.0
    assert training_size + validation_size <= 1.0
    indexes = list(range(len(data_x)))
    data_size = len(data_x)
    # seed numpy
    np.random.seed(seed=seed)
    np.random.shuffle(indexes)

    training_idx = math.ceil(data_size * training_size)
    validation_idx = math.ceil(data_size * (training_size + validation_size))

    training_set_idxs = indexes[:training_idx]
    validation_set_idxs = indexes[training_idx:validation_idx]
    test_set_idxs = indexes[validation_idx:]

    training_set_x, training_set_y = data_x[training_set_idxs], data_y[training_set_idxs]
    validation_set_x, validation_set_y = data_x[validation_set_idxs], data_y[validation_set_idxs]
    test_set_x, test_set_y = data_x[test_set_idxs], data_y[test_set_idxs]

    return training_set_x, training_set_y, \
           validation_set_x, validation_set_y, \
           test_set_x, test_set_y
           

class DataManager():
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
        
        batch_x = self.data_x[self.curr_idx:self.curr_idx+batch_size]
        batch_y = self.data_y[self.curr_idx:self.curr_idx+batch_size]
        
        
        if self.random_shift:
            random_idx = np.random.randint(low=0, high=271)
            batch_x =  (batch_x[:, :, -random_idx:], batch_x[:, :, :-random_idx])
            batch_x = np.concatenate(batch_x, axis=2)
        
        self.curr_idx = (self.curr_idx + batch_size) % self.size
        return batch_x, batch_y
    
def expanddataset(datax, datay):
    """ Expand dataset by reversing the spectra, and moving them by 50, 100, 200
    Args:
        datax: data of spectra
        datay: an array of Y labels
    Returns:
        newdatax, newdatay: expanded data and labels
    """
    
    exp_data = np.fliplr(datax)

    shifted_data = (datax[:, :, -50:], datax[:, :, :-50])
    shifted_data1 = (datax[:, :, -100:], datax[:, :, :-100])
    shifted_data2 = (datax[:, :, -200:], datax[:, :, :-200])
    shifted_data = np.concatenate(shifted_data, axis=2)
    shifted_data1 = np.concatenate(shifted_data1, axis=2)
    shifted_data2 = np.concatenate(shifted_data2, axis=2)

    newdatax = np.concatenate((datax, exp_data, shifted_data, shifted_data1, shifted_data2), axis=0)
    newdatay = np.concatenate((datay, datay, datay, datay, datay), axis=0)
    
    return newdatax, newdatay

if __name__ == '__main__':
    dataframe = pd.read_csv('Data\\nmr_photo.csv')
    data_x, data_y, labels = read_data(dataframe)

    for i in range(100):
        if data_y[i][1] == 1 and labels[i]== '0077-post-reactor1-NMR-20180422-0756':
            print(data_y[i])
            print(labels[i])

            reag= data_x[i][0]
            reag[0:100] = 0
            mix = data_x[i][1]
            mix[0:100] = 0


            plt.plot(reag, color='blue')
            plt.plot(mix, color='red')
            plt.gca().invert_xaxis()
            plt.ylim([-0.1,0.5])
            plt.show()

