import logging
import os

def setup_logging():
    """
    Set up the logging configuration.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_configuration():
    """
    Load configuration from environment variables.
    """
    config = {
        'model_path': os.getenv('MODEL_PATH', 'default_model_path'),
        'memory_path': os.getenv('MEMORY_PATH', 'default_memory_path')
    }
    return config
