"""
Calibration script to find the optimal thresholds for the AdvancedRouter in Chat99,
including separate calibration for Sonar models used in Perplexity search.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple
import logging

from app.advanced_router import AdvancedRouter
from app.config import load_config
from app.python.helpers.rag_system import RAGSystem
from dotenv import load_dotenv

load_dotenv()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_queries(file_path: str) -> List[Dict[str, str]]:
    """Load sample queries from a JSON file."""
    try:
        with open(file_path, "r") as f:
            queries = json.load(f)
        if not isinstance(queries, list) or not all(
            isinstance(q, dict) and "content" in q for q in queries
        ):
            raise ValueError("Invalid format in sample queries file")
        return queries
    except FileNotFoundError:
        logger.error(f"Sample queries file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in sample queries file: {file_path}")
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


def calibrate_thresholds(
    router: AdvancedRouter,
    queries: List[Dict[str, str]],
    target_mid_pct: float,
    target_high_pct: float,
    target_superior_pct: float,
    target_sonar_small_pct: float,
    target_sonar_medium_pct: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate the complexity thresholds for the AdvancedRouter and Sonar models.
    """
    complexities = [
        router._assess_complexity(query["content"]) for query in queries
    ]

    def count_model_calls(threshold):
        low_calls = sum(c < threshold / 2 for c in complexities)
        mid_calls = sum(
            1 for c in complexities if threshold / 2 <= c < threshold
        )
        high_calls = sum(
            1 for c in complexities if threshold <= c < threshold * 1.5
        )
        superior_calls = sum(1 for c in complexities if c >= threshold * 1.5)
        return low_calls, mid_calls, high_calls, superior_calls

    def evaluate_threshold(threshold):
        _, mid_calls, high_calls, superior_calls = count_model_calls(threshold)
        total_calls = len(queries)
        mid_pct = mid_calls / total_calls
        high_pct = high_calls / total_calls
        superior_pct = superior_calls / total_calls

        mid_diff = abs(mid_pct - target_mid_pct)
        high_diff = abs(high_pct - target_high_pct)
        superior_diff = abs(superior_pct - target_superior_pct)

        return mid_diff + high_diff + superior_diff

    # Binary search for optimal threshold
    left, right = 0.1, 1.0
    while right - left > 0.01:
        mid = (left + right) / 2
        if evaluate_threshold(mid) < evaluate_threshold(mid + 0.01):
            right = mid
        else:
            left = mid + 0.01

    optimal_threshold = (left + right) / 2

    # Calibrate Sonar model thresholds
    sonar_models = [
        (
            "llama-3-sonar-small-32k-online"
            if c < 0.3
            else (
                "llama-3-sonar-medium-32k-online"
                if c < 0.7
                else "llama-3-sonar-large-32k-online"
            )
        )
        for c in complexities
    ]
    sonar_small_calls = sum(
        1
        for model in sonar_models
        if model == "llama-3-sonar-small-32k-online"
    )
    sonar_medium_calls = sum(
        1
        for model in sonar_models
        if model == "llama-3-sonar-medium-32k-online"
    )

    total_calls = len(queries)
    actual_sonar_small_pct = sonar_small_calls / total_calls
    actual_sonar_medium_pct = sonar_medium_calls / total_calls

    sonar_small_threshold = 0.3
    sonar_medium_threshold = 0.7

    if actual_sonar_small_pct < target_sonar_small_pct:
        sonar_small_threshold = min(sonar_small_threshold * 1.1, 0.6)
    elif actual_sonar_small_pct > target_sonar_small_pct:
        sonar_small_threshold = max(sonar_small_threshold * 0.9, 0.1)

    if actual_sonar_medium_pct < target_sonar_medium_pct:
        sonar_medium_threshold = min(sonar_medium_threshold * 1.1, 0.9)
    elif actual_sonar_medium_pct > target_sonar_medium_pct:
        sonar_medium_threshold = max(
            sonar_medium_threshold * 0.9, sonar_small_threshold + 0.1
        )

    return optimal_threshold, {
        "sonar_small_threshold": sonar_small_threshold,
        "sonar_medium_threshold": sonar_medium_threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate AdvancedRouter and Sonar model thresholds for Chat99"
    )
    parser.add_argument(
        "--sample-queries",
        type=str,
        required=True,
        help="Path to JSON file containing sample queries",
    )
    parser.add_argument(
        "--mid-model-pct",
        type=float,
        default=0.3,
        help="Target percentage of queries to route to the mid-tier model (default: 0.3)",
    )
    parser.add_argument(
        "--high-model-pct",
        type=float,
        default=0.2,
        help="Target percentage of queries to route to the high-tier model (default: 0.2)",
    )
    parser.add_argument(
        "--superior-model-pct",
        type=float,
        default=0.1,
        help="Target percentage of queries to route to the superior-tier model (default: 0.1)",
    )
    parser.add_argument(
        "--sonar-small-pct",
        type=float,
        default=0.4,
        help="Target percentage of queries to route to the Sonar small model (default: 0.4)",
    )
    parser.add_argument(
        "--sonar-medium-pct",
        type=float,
        default=0.4,
        help="Target percentage of queries to route to the Sonar medium model (default: 0.4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_results.json",
        help="Output file to save calibration results (default: calibration_results.json)",
    )
    args = parser.parse_args()

    # Load sample queries
    sample_queries = load_sample_queries(args.sample_queries)

    # Create a new AdvancedRouter instance
    config = load_config()
    rag_system = RAGSystem()
    router = AdvancedRouter(
        config, None, rag_system
    )  # Pass None as agent for calibration purposes

    # Calibrate thresholds
    optimal_threshold, sonar_thresholds = calibrate_thresholds(
        router,
        sample_queries,
        args.mid_model_pct,
        args.high_model_pct,
        args.superior_model_pct,
        args.sonar_small_pct,
        args.sonar_medium_pct,
    )

    logger.info("Optimal thresholds for:")
    logger.info(f"  {args.mid_model_pct*100}% mid-tier")
    logger.info(f"  {args.high_model_pct*100}% high-tier")
    logger.info(f"  {args.superior_model_pct*100}% superior-tier model calls:")
    logger.info(f"Main threshold: {optimal_threshold}")
    logger.info(
        f"Sonar small threshold: {sonar_thresholds['sonar_small_threshold']}"
    )
    logger.info(
        f"Sonar medium threshold: {sonar_thresholds['sonar_medium_threshold']}"
    )

    # Save results
    results = {
        "mid_model_percentage": args.mid_model_pct,
        "high_model_percentage": args.high_model_pct,
        "superior_model_percentage": args.superior_model_pct,
        "sonar_small_percentage": args.sonar_small_pct,
        "sonar_medium_percentage": args.sonar_medium_pct,
        "optimal_threshold": optimal_threshold,
        "sonar_thresholds": sonar_thresholds,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Calibration results saved to {args.output}")


if __name__ == "__main__":
    main()
