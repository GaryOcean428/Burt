"""
Main entry point for the Burt application.
"""

import argparse
import logging
from dotenv import load_dotenv
from response_generator import generate_response
from utils import setup_logging, load_configuration
from advanced_router import AdvancedRouter

# Load environment variables from .env file
load_dotenv()

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Burt application.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Burt - An intelligent AI assistant")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run threshold calibration")
    parser.add_argument("--use-dynamic-routing", action="store_true",
                        help="Use dynamic routing")
    parser.add_argument("--model", type=str, default="default",
                        help="Model to use for the main agent")
    parser.add_argument("--sub-agents", type=str, nargs='*',
                        help="List of sub-agent models to use")
    return parser.parse_args()

def main() -> None:
    """
    Main function to run the Burt application.
    """
    args = parse_args()
    config = load_configuration()

    if args.calibrate:
        # Call the calibration function if calibration is requested
        calibrate()
    else:
        # Initialize the router and model manager
        if args.use_dynamic_routing:
            router = AdvancedRouter()
        else:
            router = None

        # Generate a response using the selected model and routing strategy
        response = generate_response(args.model, router)
        print(response)

if __name__ == "__main__":
    main()
