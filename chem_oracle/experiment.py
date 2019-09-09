import logging
import random
from os import path
from typing import List, Union

import numpy as np
import pandas as pd

from ms_analyze.ms import MassSpectra

from chem_oracle.probabilistic_model import NonstructuralModel, StructuralModel


def parse_nmr_filename(filename: str) -> List[str]:
    folder, filename = path.split(filename)
    name, ext = path.splitext(filename)
    parts = name.split("-")
    return [path.join(folder, part + ".csv") for part in parts]


def parse_ms_filename(filename: str) -> List[str]:
    folder, filename = path.split(filename)
    name, ext = path.splitext(filename)
    parts = name.split("-")
    return [path.join(folder, part + ".csv") for part in parts]


def ms_is_reactive(
    ms: MassSpectra, starting_materials: List[MassSpectra], max_error: float = 0.2
):
    components = ms.find_components_adaptive(
        max_error=max_error, min_components=len(starting_materials)
    )
    return len(components) > len(starting_materials)


def nmr_is_reactive(*args):
    pass


class ExperimentManager:
    def __init__(
        self,
        xlsx_file: str,
        nmr_path: str,
        ms_path: str,
        hplc_path: str,
        N_props=4,
        structural_model=True,
        seed=None,
    ):
        """Initialize ExperimentManager with given Excel workbook.
        
        Args:
            xlsx_file (str): Name of Excel workbook to read current state from.
            N_props (int): Number of abstract properties to use in probabilistic
                model
            structural_model (bool): If set to `True`, a model representing
                each compound using a structural fingerprint string is used;
                otherwise they are treated as black boxes. 
        """
        self.xlsx_file = xlsx_file
        self.nmr_path = nmr_path
        self.ms_path = ms_path
        self.hplc_path = hplc_path
        self.N_props = N_props

        # seed RNG for reproducibility
        random.seed(seed)

        # set up logging
        self.logger = logging.getLogger("experiment-manager")

        self.read_experiments()

        self.n_compounds = len(self.reagents_df["compound"].unique())

        if structural_model:
            # calculate fingerprints
            fingerprints = None
            self.model = StructuralModel(fingerprints, N_props)
        else:
            self.model = NonstructuralModel(self.n_compounds, N_props)

    def read_experiments(self):
        with pd.ExcelFile(self.xlsx_file) as reader:
            self.reagents_df = pd.read_excel(
                reader,
                sheet_name="reagents",
                dtype=(int, str, str),  # reagent_number  # reagent_name  # data_folder
            )
            self.reactions_df: pd.DataFrame = pd.read_excel(
                reader,
                sheet_name="reactions",
                dtype=(
                    int,  # compound1
                    int,  # compound2
                    int,  # compound3
                    float,  # NMR_reactivity
                    float,  # MS reactivity
                    float,  # HPLC reactivity
                    float,  # avg_expected_reactivity
                    float,  # std_expected_reactivity
                    "datetime",  # setup_time
                    int,  # reactor_number
                    str,  # data_folder
                ),
            )

    def write_experiments(self):
        with pd.ExcelWriter(self.xlsx_file) as writer:
            self.reagents_df.to_excel(writer, sheet_name="reagents")
            self.reactions_df.to_excel(writer, sheet_name="reactions")

    def update(self, n_samples=500, sampler_params=None):
        """Update expected reactivities using probabilistic model.
        
        Args:
            observation (Observation): New reactivity observation to add.
        """
        # # select reactions with at leasty observation
        # selector = (
        #     self.reactions_df["NMR_reactivity"].notna()
        #     | self.reactions_df["MS_reactivity"].notna()
        # )
        self.model.condition(self.reactions_df, n_samples, sampler_params or {})
        trace = self.model.trace
        # caculate reactivity for binary reactions
        bin_avg = 1 - np.mean(trace["bin_doesnt_react"], axis=0)
        bin_std = np.std(trace["bin_doesnt_react"], axis=0)
        # caculate reactivity for three component reactions
        tri_avg = 1 - np.mean(trace["tri_doesnt_react"], axis=0)
        tri_std = np.std(trace["tri_doesnt_react"], axis=0)

        # sync state before writing to avoid data loss
        self.read_experiments()
        df = self.reactions_df
        # update dataframe with calculated reactivities
        df.loc[
            df["compound3"] == -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack(bin_avg, bin_std)
        df.loc[
            df["compound3"] != -1,
            ["avg_expected_reactivity", "std_expected_reactivity"],
        ] = np.stack(tri_avg, tri_std)
        self.write_experiments()

    def get_spectrum(self, reagent_number: int):
        if reagent_number == -1:
            return None

    @property
    def nmr_callback(self):
        def callback(filename):
            components = parse_nmr_filename(filename)
            if len(components) > 1:  # reaction mixture — evaluate reactivity
                reactivity = nmr_is_reactive(filename, components)
                rdf = self.reactions_df
                rdf.loc[
                    (rdf["compound1"] == components[0])
                    & (rdf["compound2"] == components[1])
                    & (rdf["compound3"] == components[2]),
                    "NMR_reactivity",
                ] = reactivity
                self.update()

        return callback

    @property
    def ms_callback(self):
        def callback(filename):
            components = parse_ms_filename(filename)
            if len(components) > 1:  # reaction mixture — evaluate reactivity
                reactivity = ms_is_reactive(filename, components)
                rdf = self.reactions_df
                rdf.loc[
                    (rdf["compound1"] == components[0])
                    & (rdf["compound2"] == components[1])
                    & (rdf["compound3"] == components[2]),
                    "MS_reactivity",
                ] = reactivity
                self.write_experiments()
                self.update()
            pass

        return callback
