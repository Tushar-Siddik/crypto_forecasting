# tests/run_tests.py
import unittest
import sys
import os

from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all tests"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)