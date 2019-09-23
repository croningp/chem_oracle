import nn_model
import pandas as pd
from matplotlib import pyplot as plt


file_path = "Z:\\group\\Dario Caramelli\\Projects\\FinderX\\data\\20180418-1809-photochemical_space\\0013\\0013-post-reactor2-NMR-20180419-0545"

reagents = ["phenylhydrazine", "glycidyl_propargyl_ether"]

# data = raw_nmr_to_dataframe(file_path, reagents)
#
# data_x, data_y = read_data(data)
#
# nn = nn_model.NMR_nn()
#
# react = nn.predict(data_x)
# print(react)


print(nn_model.full_nmr_process(file_path, reagents))
