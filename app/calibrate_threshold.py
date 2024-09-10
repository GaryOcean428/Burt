"""
Calibration script to find the optimal thresholds for the AdvancedRouter in Chat99.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from app.advanced_router import AdvancedRouter
from app.config import load_config


def load_sample_queries(file_path: str) -> List[Dict[str, str]]:
    """Load sample queries from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calibrate_thresholds(
    router: AdvancedRouter,
    queries: List[Dict[str, str]],
    target_mid_pct: float,
    target_high_pct: float,
) -> Dict[str, float]:
    """
    Calibrate the complexity thresholds for the AdvancedRouter.
    """
    complexities = [router._assess_complexity(query["content"]) for query in queries]
    mid_model_calls = sum(
        1 for complexity in complexities if complexity >= router.threshold_low
    )
    high_model_calls = sum(
        1 for complexity in complexities if complexity >= router.threshold_high
    )

    actual_mid_pct = mid_model_calls / len(queries)
    actual_high_pct = high_model_calls / len(queries)

    new_threshold_low = router.threshold_low
    new_threshold_high = router.threshold_high

    if actual_mid_pct < target_mid_pct:
        new_threshold_low = max(router.threshold_low * 0.9, 0.1)
    elif actual_mid_pct > target_mid_pct:
        new_threshold_low = min(router.threshold_low * 1.1, router.threshold_high * 0.9)

    if actual_high_pct < target_high_pct:
        new_threshold_high = max(router.threshold_high * 0.9, new_threshold_low * 1.1)
    elif actual_high_pct > target_high_pct:
        new_threshold_high = min(router.threshold_high * 1.1, 0.9)

    return {"threshold_low": new_threshold_low, "threshold_high": new_threshold_high}


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate AdvancedRouter thresholds for Chat99"
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
    router = AdvancedRouter(config, None)  # Pass None as agent for calibration purposes

    # Calibrate thresholds
    thresholds = calibrate_thresholds(
        router, sample_queries, args.mid_model_pct, args.high_model_pct
    )

    print(
        f"Optimal thresholds for {args.mid_model_pct*100}% mid-tier and {args.high_model_pct*100}% high-tier model calls:"
    )
    print(f"Low threshold: {thresholds['threshold_low']}")
    print(f"High threshold: {thresholds['threshold_high']}")

    # Save results
    results = {
        "mid_model_percentage": args.mid_model_pct,
        "high_model_percentage": args.high_model_pct,
        "optimal_thresholds": thresholds,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Calibration results saved to {args.output}")


if __name__ == "__main__":
    main()
