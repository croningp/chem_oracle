import random
from typing import List, Union

import pandas as pd

from ms_analyze.ms import MassSpectra


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

class Observation:
    def __init__(self, reagent1: int, reagent2: int, reagent3: Union[int, None], reactivity: float):
        self.reagent1 = reagent1
        self.reagent2 = reagent2
        self.reagent3 = reagent3
        self.reactivity = reactivity

class ExperimentManager:
    def __init__(self, xlsx_file: str, seed=None):
        """Initialize ExperimentManager with given Excel workbook.
        
        Args:
            data_file (str): Name of Excel workbook to read current state from.
        """
        # seed RNG for reproducibility
        random.seed(seed)
        with pd.ExcelFile(xlsx_file) as reader:
            self.reagent_df = pd.read_excel(reader, sheet_name="reagents")
            self.reactions_df = pd.read_excel(reader, sheet_name="reactions")
            self.todo_df = pd.read_excel(reader, sheet_name="todo")
    
    def update(self, observation: Observation):
        """Add an observation to the reactions dataframe and update todo.
        
        Args:
            observation (Observation): New reactivity observation to add.
        """
        pass
    
    def write(self, xlsx_file: str):
        """Write updated experiment state to Excel workbook.
        
        Args:
            xlsx_file (str): Name of Excel workbook to commit results to.
        """
        with pd.ExcelWriter(xlsx_file) as writer:
            self.reagent_df.to_excel(writer, sheet_name="reagents")
            self.reactions_df.to_excel(writer, sheet_name="reactions")
            self.todo_df.to_excel(writer, sheet_name="todo")