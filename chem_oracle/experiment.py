import logging
import random
from datetime import datetime
from os import path
from shutil import copyfile
from typing import List, Union

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from chem_oracle.probabilistic_model import NonstructuralModel, StructuralModel
from chem_oracle.util import morgan_matrix
from ms_analyze.ms import MassSpectra
from nmr_analyze.nn_model import full_nmr_process


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

        self.n_compounds = len(self.reagents_df["reagent_number"].unique())

        if structural_model:
            # calculate fingerprints
            mols = [MolFromSmiles(smiles) for smiles in self.reagents_df["SMILES"]]
            # TODO: expose this as a parameter
            fingerprints = morgan_matrix(mols, radius=3, nbits=128)
            self.model = StructuralModel(fingerprints, N_props)
        else:
            self.model = NonstructuralModel(self.n_compounds, N_props)

    def read_experiments(self):
        with pd.ExcelFile(self.xlsx_file) as reader:
            self.reagents_df: pd.DataFrame = pd.read_excel(
                reader,
                sheet_name="reagents",
                dtype={
                    "reagent_number": int,
                    "CAS_number": str,
                    "reagent_name": str,
                    "SMILES": str,
                    "data_folder": str,
                },
            )
            self.reactions_df: pd.DataFrame = pd.read_excel(
                reader,
                sheet_name="reactions",
                dtype={
                    "compound1": int,
                    "compound2": int,
                    "compound3": int,
                    "NMR_reactivity": float,
                    "MS reactivity": float,
                    "HPLC reactivity": float,
                    "avg_expected_reactivity": float,
                    "std_expected_reactivity": float,
                    "setup_time": str,
                    # to support NaN
                    "reactor_number": float,
                    "data_folder": str,
                },
                parse_dates=["setup_time"],
            )

    def write_experiments(self, backup=True):
        if backup and path.exists(self.xlsx_file):
            timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-")
            dst_file, ext = path.splitext(self.xlsx_file)
            dst_file = dst_file + timestamp + ext
            copyfile(self.xlsx_file, dst_file)
        with pd.ExcelWriter(self.xlsx_file) as writer:
            self.reagents_df.to_excel(writer, sheet_name="reagents", index=False)
            self.reactions_df.to_excel(writer, sheet_name="reactions", index=False)

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

    def populate(self):
        """
        Add entries for missing reactions to reaction dataframe.
        """
        all_compounds = self.reagents_df["reagent_number"]
        n_compounds = len(all_compounds)
        df = self.reactions_df
        idx = len(df)
        for i1, c1 in enumerate(all_compounds):
            for i2, c2 in enumerate(all_compounds):
                if i2 <= i1:
                    continue
                for c3 in list(all_compounds[max(i1, i2) + 1 :]) + [-1]:
                    # check whether entry already exists
                    if df.query(
                        "compound1 == @c1 and compound2 == @c2 and compound3 == @c3"
                    ).empty:
                        # add missing entry and increment index
                        self.reactions_df.loc[idx] = (
                            c1,  # compound1
                            c2,  # compound2
                            c3,  # compound3
                            None,  # NMR_reactivity
                            None,  # MS_reactivity
                            None,  # HPLC_reactivity
                            None,  # avg_expected_reactivity
                            None,  # std_expected_reactivity
                            None,  # setup_time
                            None,  # reactor_number
                            None,  # data_folder
                        )
                        idx += 1
