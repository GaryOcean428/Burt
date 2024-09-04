import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_abs_path(relative_path):
    """
    Convert a relative path to an absolute path.

    Args:
    relative_path (str): The relative path to convert.

    Returns:
    str: The absolute path.

    Raises:
    ValueError: If the relative_path is empty or None.
    """
    if not relative_path:
        logger.error("Empty or None relative path provided")
        raise ValueError("Relative path cannot be empty or None")

    try:
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', relative_path))
        logger.info(f"Converted relative path '{relative_path}' to absolute path '{abs_path}'")
        return abs_path
    except Exception as e:
        logger.error(f"Error converting relative path '{relative_path}' to absolute path: {str(e)}")
        raise

def read_file(file_path):
    """
    Read the contents of a file.

    Args:
    file_path (str): The path to the file to read.

    Returns:
    str: The contents of the file.

    Raises:
    FileNotFoundError: If the file does not exist.
    IOError: If there's an error reading the file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except IOError as e:
        logger.error(f"Error reading file '{file_path}': {str(e)}")
        raise

# Add more file-related functions as needed
