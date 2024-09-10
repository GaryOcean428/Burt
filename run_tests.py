import unittest
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from app.tests.test_online_knowledge_tool import TestOnlineKnowledgeTool
from app.tests.test_vector_db import TestVectorDB

if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOnlineKnowledgeTool))
    suite.addTest(unittest.makeSuite(TestVectorDB))

    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Print the result
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")

    # Print any errors or failures
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"{test}: {error}")

    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"{test}: {failure}")

    # Set exit code based on test results
    sys.exit(not result.wasSuccessful())
