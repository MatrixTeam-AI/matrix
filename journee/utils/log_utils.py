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
    print(f"{logfire_token=}")
    logfire.configure(token=logfire_token)
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

    # Ensure we display INFO logs.
    logging.basicConfig(level=logging.INFO)
    # Create our standard logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()
logger_info = logger.info