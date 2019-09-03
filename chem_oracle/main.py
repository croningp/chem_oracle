import time
import logging

from watchdog.observers import Observer

from chem_oracle.monitoring import MSEventHandler, NMREventHandler

def main():
    # set up logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("chem_oracle")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # set up file system monitors
    nmr_handler = NMREventHandler()
    ms_handler = MSEventHandler()
    observer = Observer()
    observer.schedule(nmr_handler, ".", recursive=False)
    observer.schedule(ms_handler, ".", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()