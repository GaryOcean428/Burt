import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from app.advanced_router import AdvancedRouter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a simple configuration for testing
test_config = {
    "ROUTER_THRESHOLD": 0.7,
    "rate_limit_requests": 120,
    "rate_limit_input_tokens": 200000,
    "rate_limit_output_tokens": 200000,
    "rate_limit_seconds": 60,
}

# Initialize the AdvancedRouter
router = AdvancedRouter(test_config)

# Test query
test_query = "Analyze the impact of artificial intelligence on job markets in the next decade."

try:
    # Process the query
    result = router.process_query_advanced(test_query)

    # Print the result
    print("Query processed successfully:")
    print(f"Selected model: {result['model']}")
    print(f"Task type: {result['task_type']}")
    print(f"Task complexity: {result['task_complexity']}")
    print(f"Parameters: {result['params']}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback

    traceback.print_exc()
