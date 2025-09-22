import unittest
import sys
import importlib.util
from pathlib import Path


def run_all_tests():
    """Discover and run all tests in the tests directory"""

    # Set up import paths - run from project root for proper relative imports
    project_root = Path(__file__).parent.parent.parent
    src_dir = Path(__file__).parent.parent

    # Add project root and src to path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_dir))

    # Use unittest's built-in discovery which handles imports better
    loader = unittest.TestLoader()

    # Change to project root directory for proper relative imports
    import os

    original_cwd = os.getcwd()
    os.chdir(str(project_root))

    try:
        # Discover all tests in the src/tests directory
        suite = loader.discover("src.tests", pattern="test_*.py")

        print(f"Discovered {suite.countTestCases()} test cases")

        # Get list of test files for reporting
        test_files = list(Path("src/tests").glob("test_*.py"))
        print(f"Found {len(test_files)} test files")

    except Exception as e:
        print(f"Error during test discovery: {e}")
        suite = unittest.TestSuite()

    finally:
        # Change back to original directory
        os.chdir(original_cwd)

    print(f"\nRunning {suite.countTestCases()} tests...\n")

    # Change to project root directory for test execution
    os.chdir(str(project_root))

    try:
        # Run the tests
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(suite)
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

    # Print summary
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")

    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("SOME TESTS FAILED!")
        print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        if result.failures:
            print(f"\nFAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print(f"\nERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")

    print(f"\nTotal tests run: {result.testsRun}")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
