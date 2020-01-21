import glob
import logging
import os
import random
import threading
import time
from datetime import datetime
from os import path
from shutil import copyfile
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from chem_oracle import util
from chem_oracle.probabilistic_model import NonstructuralModel, StructuralModel
from chem_oracle.util import morgan_matrix
from hplc_analyze.hplc_reactivity import hplc_process
from ms_analyze.ms import MassSpectra, MassSpectrum
from nmr_analyze.nn_model import full_nmr_process


def nmr_is_reactive(experiment_dir: str, starting_material_dirs: List[str]) -> bool:
    data_dir = path.dirname(experiment_dir)
    return full_nmr_process(experiment_dir, starting_material_dirs, data_dir) > 0.5


def match(ms: MassSpectrum, ms_ref: MassSpectrum, mass_tol=1.0):
    """
    Tries to match every peak in `ms` with one in `ms_ref` at most
    `mass_tol` apart.
    :param ms: Sample mass spectrum
    :param ms_ref: Reference mass spectrum, assuming the same mass axis as sample spectrum
    :param mass_tol: Maximum distance between two peaks (in
    :return: `True` if all peaks in `ms` can be matched, `False` otherwise.
    """
    peaks = ms.find_peaks(height=0.1)
    peaks_ref = ms_ref.find_peaks(height=0.025)
    diff = np.abs(peaks - peaks_ref[:, np.newaxis]) < np.round(
        mass_tol / ms.mass_resolution
    )
    return diff.any(axis=0).all()


def ms_is_reactive(
    spectrum_dir: str,
    starting_material_dirs: str,
    max_error: float = 0.1,
    mass_resolution: float = 1.0,
    ms_file_suffix: str = "_is1",
):
    rxn_file = glob.glob(path.join(spectrum_dir, f"*{ms_file_suffix}.npz"))[0]
    reactant_files = [
        glob.glob(path.join(d, f"*{ms_file_suffix}.npz"))[0]
        for d in starting_material_dirs
    ]
    rxn_ms = MassSpectra.from_npz(rxn_file).discretize(mass_resolution)
    reactant_ms = [
        MassSpectra.from_npz(ms_file).discretize(mass_resolution)
        for ms_file in reactant_files
    ]
    # sum up all reactant chromatograms
    for r_ms in reactant_ms[1:]:
        reactant_ms[0].spectra += r_ms.spectra

    rxn_components = rxn_ms.find_components_adaptive(max_error)[1]
    reactant_components = reactant_ms[0].find_components_adaptive(max_error)[1]

    for component in rxn_components:
        new = True
        for component_ref in reactant_components:
            if match(component, component_ref):
                new = False
        if new:
            return True
    return False


class ExperimentManager:
    def __init__(
        self,
        xlsx_file: str,
        N_props=4,
        structural_model=True,
        fingerprint_radius=1,
        fingerprint_bits=256,
        seed=None,
    ):
        """
        Initialize ExperimentManager with given Excel workbook.
        
        Args:
            xlsx_file (str): Name of Excel workbook to read current state from.
                Experiment data files are expected to be in the same folder.
            N_props (int): Number of abstract properties to use in probabilistic
                model
            structural_model (bool): If set to `True`, a model representing
                each compound using a structural fingerprint string is used;
                otherwise they are treated as black boxes.
        """
        self.xlsx_file = xlsx_file
        self.N_props = N_props
        self.data_dir = path.dirname(self.xlsx_file)
        self.reagents_dir = path.join(self.data_dir, "reagents")
        self.update_lock = threading.Lock()
        self.should_update = False

        # seed RNG for reproducibility
        random.seed(seed)

        # set up logging
        self.logger = logging.getLogger("experiment-manager")
        self.logger.setLevel(logging.DEBUG)

        self.read_experiments()

        self.n_compounds = len(self.reagents_df["reagent_number"].unique())

        if structural_model:
            # calculate fingerprints
            self.mols = [MolFromSmiles(smiles) for smiles in self.reagents_df["SMILES"]]
            # TODO: expose this as a parameter
            self.fingerprints = morgan_matrix(
                self.mols, radius=fingerprint_radius, nbits=fingerprint_bits
            )
            self.model = StructuralModel(self.fingerprints, N_props)
        else:
            self.model = NonstructuralModel(self.n_compounds, N_props)

        # start update loop
        threading.Thread(target=self.update_loop, daemon=True).start()

    def read_experiments(self):
        with pd.ExcelFile(self.xlsx_file) as reader:
            self.reagents_df: pd.DataFrame = pd.read_excel(
                reader,
                sheet_name="reagents",
                dtype={
                    "reagent_number": int,
                    "CAS_number": str,
                    "reagent_name": str,
                    "flask_name": str,
                    "SMILES": str,
                    "data_folder": str,
                },
            )
            self.reactions_df: pd.DataFrame = pd.read_excel(
                reader,
                sheet_name="reactions",
                dtype={
                    # to support NaN
                    "reaction_number": float,
                    "compound1": int,
                    "compound2": int,
                    "compound3": int,
                    "NMR_reactivity": float,
                    "MS reactivity": float,
                    "HPLC reactivity": float,
                    "avg_expected_reactivity": float,
                    "std_expected_reactivity": float,
                    "reactivity_disruption": float,
                    "uncertainty_disruption": float,
                },
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

    def update_loop(self, backup=True, **params):
        while True:
            if self.should_update:
                self.update(**params)
                self.write_experiments(backup)
                self.should_update = False
            time.sleep(30)

    def update(self, n_samples=250, variational=False, **pymc3_params):
        """Update expected reactivities using probabilistic model.
        
        Args:
            n_samples (int): Number of samples in each MCMC chain.
        """
        with self.update_lock:
            self.model.sample(self.reactions_df, n_samples, variational, **pymc3_params)
            self.reactions_df = self.model.condition(self.reactions_df)

    def data_folder(self, reagent_number: int, data_type: str, blank=False) -> str:
        """
        Gives the full path to the data folder for a given reagent number and
        data type.
        :param reagent_number: TODO
        :param data_type: "HPLC", "MS", or "NMR"
        :param blank: Return the corresponding blank experiment.
        :return: Full path to reagent data of requested type.
        """
        suffixes = {"HPLC": "_HPLC", "MS": "_MS", "NMR": "_1H"}
        suffix = suffixes[data_type]
        for p in os.listdir(self.reagents_dir):
            if suffix not in p:
                continue
            if blank and "BLANK" not in p:
                continue
            if not blank and "BLANK" in p:
                continue
            if util.reaction_components(p) == [reagent_number]:
                return path.join(self.reagents_dir, p)
        raise Exception(f"{data_type} folder for reagent {reagent_number} not found.")

    def hplc_callback(self, data_dir: str):
        self.logger.info(f"HPLC path {data_dir} - detected.")
        time.sleep(1.0)
        with self.update_lock:
            self.add_data(data_dir, data_type="HPLC")
        self.should_update = True

    def ms_callback(self, data_dir: str):
        self.logger.info(f"MS path {data_dir} - detected.")
        time.sleep(1.0)
        with self.update_lock:
            self.add_data(data_dir, data_type="MS")
        self.should_update = True

    def nmr_callback(self, data_dir: str):
        self.logger.info(f"Proton NMR path {data_dir} - detected.")
        time.sleep(1.0)
        with self.update_lock:
            self.add_data(data_dir, data_type="NMR")
        self.should_update = True

    def find_reaction(self, components):
        rdf = self.reactions_df
        return (
            (rdf["compound1"] == components[0])
            & (rdf["compound2"] == components[1])
            & (rdf["compound3"] == (components[2] if len(components) == 3 else -1))
        )

    def add_data(self, data_dir: str, data_type: str, override: bool = False, **params):
        if "BLANK" in data_dir:
            return
        reaction_number = util.reaction_number(data_dir)
        components = util.reaction_components(data_dir)
        reactivity_column = f"{data_type}_reactivity"

        # skip if reactivity data already exists
        if (
            not override
            and (
                self.reactions_df[reactivity_column].notna()
                & self.find_reaction(components)
                & (self.reactions_df["reaction_number"] == reaction_number)
            ).any()
        ):
            self.logger.info(
                f"{data_type} data for reaction {reaction_number} between "
                f"{components} already processed - skipping."
            )
            return

        if len(components) > 1:  # reaction mixture — evaluate reactivity
            self.logger.info(
                f"Adding {data_type} data for reaction {reaction_number}: {components}."
            )
            if data_type == "MS":
                component_dirs = [
                    self.data_folder(component, data_type="MS")
                    for component in components
                ]
                reactivity = ms_is_reactive(data_dir, component_dirs, **params)
            elif data_type == "HPLC":
                reactivity = hplc_process(data_dir)
            elif data_type == "NMR":
                # TODO: Assess reactivity
                return
            self.logger.info(f"{data_type} reactivity: {reactivity}")
            rdf = self.reactions_df
            selector = self.find_reaction(components)
            rdf.loc[selector, "reaction_number"] = reaction_number
            rdf.loc[selector, reactivity_column] = reactivity


    def populate(self):
        """
        Add entries for missing reactions to reaction dataframe.
        Existing entries are left intact.
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
                            None,  # reactor_number
                            c1,  # compound1
                            c2,  # compound2
                            c3,  # compound3
                            None,  # NMR_reactivity
                            None,  # MS_reactivity
                            None,  # HPLC_reactivity
                            None,  # avg_expected_reactivity
                            None,  # std_expected_reactivity
                        )
                        idx += 1
        # convert compound numbers back to int (pandas bug)
        self.reactions_df = df.astype(
            {"compound1": "int", "compound2": "int", "compound3": "int"}
        )
