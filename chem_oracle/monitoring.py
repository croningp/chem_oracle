import logging
from collections import namedtuple

from watchdog.events import FileSystemEventHandler
from os import path
from ms_analyze.ms import MassSpectra
from typing import NamedTuple, Callable


def parse_filename(filename: str):
    # default value
    compounds = [-1, -1, -1]
    filename = path.basename(filename)
    for i, part in enumerate(filename.split("-")):
        compounds[i] = int(part)
    return compounds


class DataEventHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str]]):
        super().__init__()
        self.logger = logging.getLogger("chem_oracle.NMR_monitor")

    def on_any_event(self, event):
        super().on_any_event(event)
        self.logger.debug(f"NMR event {event}.")
        filename = ""
        nmr_spectrum = None
        data_label = DataLabel(filename)
        self.callback(filename)
