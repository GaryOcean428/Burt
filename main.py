"""
Main entry point for the Chat99 application.
"""

import argparse
import logging
from dotenv import load_dotenv
from chat99 import chat_with_99
from calibrate_threshold import main as calibrate
from utils import setup_logging, load_configuration
from config import DEFAULT_ROUTER, DEFAULT_THRESHOLD

# Load environment variables from .env file
load_dotenv()

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Chat99 application.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Chat99 - An intelligent AI assistant")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run threshold calibration")
    parser.add_argument("--use-dynamic-routing", action="store_true",
                        help="Use dynamic routing")
    parser.add_argument("--router", type=str, default=DEFAULT_ROUTER,
                        help=f"Router to use (default: {DEFAULT_ROUTER})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Routing threshold (default: {DEFAULT_THRESHOLD})")
    return parser.parse_args()

def main() -> None:
    """
    Main function to run the Chat99 application.
    """
    setup_logging()
    config = load_configuration()
    args = parse_args()
    
    if args.calibrate:
        calibrate()
    else:
        chat_with_99(args, config)

if __name__ == "__main__":
    main()
