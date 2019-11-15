import logging
import os
import sys
from os import path

from watchdog.observers.polling import PollingObserver as Observer

from chem_oracle.experiment import ExperimentManager
from chem_oracle.monitoring import DataEventHandler


def main(xlsx_file: str, N_props=4, structural_model=True):
    global manager
    xlsx_file = path.abspath(xlsx_file)
    data_dir = path.dirname(xlsx_file)
    manager = ExperimentManager(xlsx_file, N_props, structural_model)
    # set up logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("chem_oracle")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logging.getLogger("experiment-manager").addHandler(handler)

    # set up file system monitors
    nmr_handler = DataEventHandler(manager.add_nmr, patterns=["*_1H"])
    ms_handler = DataEventHandler(manager.add_ms, patterns=["*_MS"])
    hplc_hander = DataEventHandler(manager.add_hplc, patterns=["*_HPLC"])
    observer = Observer()
    observer.schedule(nmr_handler, data_dir, recursive=True)
    observer.schedule(ms_handler, data_dir, recursive=True)
    observer.schedule(hplc_hander, data_dir, recursive=True)
    observer.start()

    # scan data dir for existing data
    for folder in os.listdir(data_dir):
        full_path = path.join(data_dir, folder)
        if folder.endswith("_1H"):
            manager.add_nmr(full_path)
        elif folder.endswith("_MS"):
            manager.add_ms(full_path)
        elif folder.endswith("_HPLC"):
            manager.add_hplc(full_path)


if __name__ == "__main__":
    manager = None
    main(sys.argv[1])
