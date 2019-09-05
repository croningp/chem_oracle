import logging
import random
from typing import List, Union

import pandas as pd

from ms_analyze.ms import MassSpectra
from .probabilistic_model import StructuralModel, NonstructuralModel
from chem_oracle.monitoring import parse_filename


class Experiment:
    def __init__(self, starting_materials: List[int], ms: MassSpectra = None, nmr=None):
        self.starting_materials = starting_materials
        self.ms = ms
        self.nmr = nmr

    def is_reactive(self, max_error: float = 0.2):
        components = self.ms.find_components_adaptive(
            max_error=max_error, min_components=len(self.starting_materials)
        )
        return len(components) > len(self.starting_materials)


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
        self.nmr_path = nmr_path
        self.ms_path = ms_path
        self.hplc_path = hplc_path
        self.N_props = N_props

        # seed RNG for reproducibility
        random.seed(seed)

        # set up logging
        self.logger = logging.getLogger("experiment-manager")

        with pd.ExcelFile(xlsx_file) as reader:
            self.reagent_df = pd.read_excel(
                reader, sheet_name="reagents", dtype=(int, str)
            )
            self.reactions_df = pd.read_excel(
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
            self.todo_df = pd.read_excel(reader, sheet_name="todo")

        self.n_compounds = len(self.reagent_df["compound"].unique())

        if structural_model:
            # calculate fingerprints
            fingerprints = None
            self.model = StructuralModel(fingerprints, N_props)
        else:
            self.model = NonstructuralModel(self.n_compounds, N_props)

    def update(self, n_samples=500, sampler_params=None, write=True):
        """Update expected reactivities using probabilistic model.
        
        Args:
            observation (Observation): New reactivity observation to add.
        """
        self.model.condition(self.reactions_df, n_samples, sampler_params or {})
        # TODO: update dataframe from trace
        if write:
            new_xlsx_file = ""
            self.write(new_xlsx_file)

    def write(self, xlsx_file: str):
        """Write updated experiment state to Excel workbook.
        
        Args:
            xlsx_file (str): Name of Excel workbook to commit results to.
        """
        with pd.ExcelWriter(xlsx_file) as writer:
            self.reagent_df.to_excel(writer, sheet_name="reagents")
            self.reactions_df.to_excel(writer, sheet_name="reactions")
            self.todo_df.to_excel(writer, sheet_name="todo")

    def get_spectrum(self, reagent_number: int):
        if reagent_number == -1:
            return None

    @property
    def nmr_callback(self):
        def callback(filename):
            # 1. lookup the reaction (look up the spectra for starting materials and reaction mixture)
            spectrum = NMRSpectrum(filename)
            components = parse_filename(filename)
            try:
                component_spectra = [
                    self.get_spectrum(component) for component in components
                ]
            except FileNotFoundError as e:
                self.logger.warn(
                    f"{e}\nCould not open component spectra; model update aborted."
                )
                return
            reactivity = nmr_reactivity(spectrum, component_spectra)
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
            pass

        return callback
