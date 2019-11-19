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

from chem_oracle import chemstation
from chem_oracle.probabilistic_model import NonstructuralModel, StructuralModel
from chem_oracle.util import morgan_matrix
from ms_analyze.ms import MassSpectra
from nmr_analyze.nn_model import full_nmr_process


def reaction_number(experiment_dir: str) -> int:
    experiment_dir = path.basename(experiment_dir)
    return int(experiment_dir.split("_")[0])


def reaction_components(experiment_dir: str) -> List[int]:
    experiment_dir = path.basename(experiment_dir)
    components = experiment_dir.split("_")[1]
    return [int(s) for s in components.split("-")]


def nmr_is_reactive(experiment_dir: str, starting_material_dirs: List[str]) -> bool:
    data_dir = path.dirname(experiment_dir)
    return full_nmr_process(experiment_dir, starting_material_dirs, data_dir) > 0.5


def read_hplc(experiment_dir: str, channel: str = "A"):
    filename = path.join(experiment_dir, f"DAD1{channel}")
    npz_file = filename + ".npz"
    if path.exists(npz_file):
        # already processed
        data = np.load(npz_file)
        return data["times"], data["values"]
    else:
        print(f"Not found {npz_file}")
        ch_file = filename + ".ch"
        data = chemstation.CHFile(ch_file)
        np.savez_compressed(npz_file, times=data.times, values=data.values)
        return np.array(data.times), np.array(data.values)


def hplc_is_reactive(experiment_dir: str, starting_materials_dirs: List[str]) -> bool:
    product_spectrum = read_hplc(experiment_dir)
    starting_material_spectra = [read_hplc(sm) for sm in starting_materials_dirs]

    return None


def ms_is_reactive(
    spectrum_dir: str,
    starting_materials,
    max_error: float = 0.2,
    ms_file_suffix: str = "_is1",
):
    ms_file = glob.glob(path.join(spectrum_dir, f"*{ms_file_suffix}.npz"))[0]
    ms = MassSpectra.from_npz(ms_file)
    components = ms.find_components_adaptive(
        max_error=max_error, min_components=len(starting_materials)
    )
    return len(components[1]) > len(starting_materials)


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
        self.model.condition(self.reactions_df, n_samples, variational, **pymc3_params)
        trace = self.model.trace
        # caculate reactivity for binary reactions
        bin_avg = 1 - np.mean(trace["bin_doesnt_react"], axis=0)
        bin_std = np.std(trace["bin_doesnt_react"], axis=0)
        # caculate reactivity for three component reactions
        tri_avg = 1 - np.mean(trace["tri_doesnt_react"], axis=0)
        tri_std = np.std(trace["tri_doesnt_react"], axis=0)

        with self.update_lock:
            df = self.reactions_df
            # update dataframe with calculated reactivities
            df.loc[
                df["compound3"] == -1,
                ["avg_expected_reactivity", "std_expected_reactivity"],
            ] = np.stack([bin_avg, bin_std]).T
            df.loc[
                df["compound3"] != -1,
                ["avg_expected_reactivity", "std_expected_reactivity"],
            ] = np.stack([tri_avg, tri_std]).T

    def nmr_folder(self, reagent_number: int) -> str:
        for p in os.listdir(self.reagents_dir):
            if "_1H" not in p:
                continue
            if reaction_components(p) == [reagent_number]:
                return path.join(self.reagents_dir, p)
        raise Exception(f"NMR spectrum for reagent {reagent_number} not found.")

    def hplc_folder(self, reagent_number: int) -> str:
        for p in os.listdir(self.reagents_dir):
            if "_HPLC" not in p or "BLANK" in p:
                continue
            if reaction_components(p) == [reagent_number]:
                return path.join(self.reagents_dir, p)
        raise Exception(f"HPLC for {reagent_number} not found.")

    def hplc_callback(self, data_dir: str):
        self.logger.info(f"HPLC path {data_dir} - detected.")
        with self.update_lock:
            self.add_hplc(data_dir)
        self.should_update = True

    def ms_callback(self, data_dir: str):
        self.logger.info(f"MS path {data_dir} - detected.")
        with self.update_lock:
            self.add_ms(data_dir)
        self.should_update = True

    def nmr_callback(self, data_dir: str):
        self.logger.info(f"Proton NMR path {data_dir} - detected.")
        with self.update_lock:
            self.add_nmr(data_dir)
        self.should_update = True

    def find_reaction(self, components):
        rdf = self.reactions_df
        return (
            (rdf["compound1"] == components[0])
            & (rdf["compound2"] == components[1])
            & (rdf["compound3"] == (components[2] if len(components) == 3 else -1))
        )

    def add_nmr(self, data_dir: str):
        components = reaction_components(data_dir)
        if len(components) > 1:  # reaction mixture — evaluate reactivity
            self.logger.info(f"Adding NMR spectrum for reaction {components}.")
            component_paths = [self.nmr_folder(component) for component in components]
            reactivity = nmr_is_reactive(data_dir, component_paths)
            rdf = self.reactions_df
            selector = self.find_reaction(components)
            rdf.loc[selector, "reaction_number"] = reaction_number(data_dir)
            rdf.loc[selector, "NMR_reactivity"] = reactivity

    def add_ms(self, data_dir: str):
        if "BLANK" in data_dir:
            return
        components = reaction_components(data_dir)
        if len(components) > 1:  # reaction mixture — evaluate reactivity
            self.logger.info(f"Adding MS spectrum for reaction {components}.")
            reactivity = ms_is_reactive(data_dir, components, max_error=0.15)
            rdf = self.reactions_df
            selector = self.find_reaction(components)
            rdf.loc[selector, "reaction_number"] = reaction_number(data_dir)
            rdf.loc[selector, "MS_reactivity"] = reactivity

    def add_hplc(self, data_dir: str):
        if "BLANK" in data_dir:
            return
        components = reaction_components(data_dir)
        if len(components) > 1:  # reaction mixture — evaluate reactivity
            self.logger.info(f"Adding HPLC spectrum for reaction {components}.")
            component_paths = map(self.hplc_folder, components)
            reactivity = hplc_is_reactive(data_dir, component_paths)
            rdf = self.reactions_df
            selector = self.find_reaction(components)
            rdf.loc[selector, "reaction_number"] = reaction_number(data_dir)
            rdf.loc[selector, "HPLC_reactivity"] = reactivity

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
