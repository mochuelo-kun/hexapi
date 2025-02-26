import logging
from datetime import datetime

class HumanTimestampFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def setup_logging(level=logging.INFO):
    """Configure logging with human-readable timestamps"""
    formatter = HumanTimestampFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger('chrysalis')
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    return root_logger 