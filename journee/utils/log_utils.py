import os
import logging
import logfire

def read_logfire_token(file_path):
    try:
        with open(file_path, 'r') as file:
            token = file.read().strip()
        return token
    except Exception as e:
        print(f"ReadTokenError: {e}")
        return "Dummy_Token"
    
def setup_logging():
    # Configure Logfire to send logs to the Logfire service.
    # This is optional, but very helpful for debugging deployed applications.
    logfire_token_file = os.path.join(
        os.path.dirname(__file__), 
        'logfire_token.txt'
    )
    logfire_token = read_logfire_token(logfire_token_file)
    logfire.configure(token=logfire_token)
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

    # Ensure we display INFO logs.
    logging.basicConfig(level=logging.INFO)
    # Create our standard logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

class StdWrapper:
    def __init__(self, original_std, logger, level):
        self.original_std = original_std
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.rstrip() != "":
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        if self.buffer.rstrip() != "":
            self.logger.log(self.level, self.buffer.rstrip())
            self.buffer = ""

    def __getattr__(self, attr):
        return getattr(self.original_std, attr)

def redirect_stdout_err_to_logger(logger):
    import sys
    sys.stdout = StdWrapper(sys.stdout, logger, logging.INFO)
    sys.stderr = StdWrapper(sys.stderr, logger, logging.ERROR)

logger = setup_logging()