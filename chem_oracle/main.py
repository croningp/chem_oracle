import logging
import os
import sys
from os import path

from watchdog.observers.polling import PollingObserver as Observer

from chem_oracle.experiment import ExperimentManager
from chem_oracle.monitoring import DataEventHandler


def main(manager: ExperimentManager):
    xlsx_file = path.abspath(manager.xlsx_file)
    data_dir = path.dirname(xlsx_file)
    # set up logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("chem_oracle")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logging.getLogger("experiment-manager").addHandler(handler)

    # set up file system monitors
    nmr_handler = DataEventHandler(manager.nmr_callback, patterns=["*_1H"])
    ms_handler = DataEventHandler(manager.ms_callback, patterns=["*_MS"])
    hplc_hander = DataEventHandler(manager.hplc_callback, patterns=["*_HPLC"])
    observer = Observer()
    observer.schedule(nmr_handler, path=data_dir, recursive=False)
    observer.schedule(ms_handler, path=data_dir, recursive=False)
    observer.schedule(hplc_hander, path=data_dir, recursive=False)
    observer.start()

    # scan data dir for existing data
    for folder in os.listdir(data_dir):
        full_path = path.join(data_dir, folder)
        if folder.endswith("_1H"):
            manager.add_data(full_path, data_type="NMR")
        elif folder.endswith("_MS"):
            manager.add_data(full_path, data_type="MS")
        elif folder.endswith("_HPLC"):
            manager.add_data(full_path, data_type="HPLC")


if __name__ == "__main__":
    mgr = ExperimentManager(sys.argv[-1])
    mgr = main(mgr)
