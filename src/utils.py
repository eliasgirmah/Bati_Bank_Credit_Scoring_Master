import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up as many loggers as you want."""
    
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler
    handler = logging.FileHandler(log_file)        
    
    # Create logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    # Logs will be saved inside the logs directory
    os.makedirs("../logs", exist_ok=True)
    log_file = "../logs/data_preprocessing.log"
    
    # Set up logger
    logger = setup_logger("data_preprocessing", log_file)
    
    # Example log messages
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
