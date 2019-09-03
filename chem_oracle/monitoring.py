import logging
from watchdog.events import FileSystemEventHandler


class NMREventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("chem_oracle.NMR_monitor")
    
    def on_any_event(self, event):
        super().on_any_event(event)
        self.logger.debug(f"NMR event {event}.")


class MSEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("chem_oracle.MS_monitor")
    
    def on_any_event(self, event):
        super().on_any_event(event)
        self.logger.debug(f"Mass spec event {event}.")
