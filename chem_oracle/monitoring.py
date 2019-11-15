import logging
from os import path
from typing import Callable, Any

from watchdog.events import PatternMatchingEventHandler


def parse_filename(filename: str):
    # default value
    compounds = [-1, -1, -1]
    filename = path.basename(filename)
    for i, part in enumerate(filename.split("-")):
        compounds[i] = int(part)
    return compounds


class DataEventHandler(PatternMatchingEventHandler):
    def __init__(self, callback: Callable[[str], Any], **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.logger = logging.getLogger("chem_oracle.NMR_monitor")

    def on_created(self, event):
        super().on_created(event)
        self.logger.debug(f"Filesystem event {event}.")
        self.callback(event.src_path)
