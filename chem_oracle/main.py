import time
import sys
import logging

from watchdog.observers import Observer

from chem_oracle.monitoring import DataEventHandler
from chem_oracle.experiment import ExperimentManager


def main(xlsx_file: str, N_props=4, structural_model=True):
    manager = ExperimentManager(xlsx_file, N_props, structural_model)
    # set up logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("chem_oracle")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # set up file system monitors
    nmr_handler = DataEventHandler(manager.nmr_callback, patterns=["**/data.1d"])
    ms_handler = DataEventHandler(manager.ms_callback, patterns=["*.datx"])
    observer = Observer()
    observer.schedule(nmr_handler, ".", recursive=True)
    observer.schedule(ms_handler, ".", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main(sys.argv[1])
