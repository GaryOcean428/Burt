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
from models import ModelManager, SubAgent
from advanced_router import AdvancedRouter

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
    parser.add_argument("--model", type=str, default="default",
                        help="Model to use for the main agent")
    parser.add_argument("--sub-agents", type=str, nargs='*',
                        help="List of sub-agent models to use")
    return parser.parse_args()

def initialize_components(config, args):
    """
    Initialize and return the main components: memory, model manager, and agent.
    """
    logger.info("Initializing memory")
    memory = Memory(config['memory_path'])
    
    logger.info(f"Initializing model manager with model {args.model}")
    model_manager = ModelManager(model_name=args.model)
    
    logger.info("Initializing sub-agents")
    sub_agents = [SubAgent(model_name=sub_agent) for sub_agent in args.sub_agents or []]
    
    logger.info("Initializing agent")
    agent = Agent(memory, model_manager, sub_agents)

    return memory, model_manager, agent

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
        memory, model_manager, agent = initialize_components(config, args)
        
        if args.use_dynamic_routing:
            logger.info("Using dynamic routing")
            advanced_router = AdvancedRouter()
            agent.router = advanced_router
        
        chat_with_99(args, config, agent)

if __name__ == "__main__":
    main()
