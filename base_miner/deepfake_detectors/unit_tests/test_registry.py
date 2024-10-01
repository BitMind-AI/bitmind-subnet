import unittest
import os
import sys

#Registry class located in the parent directory
directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(directory)
sys.path.append(parent_directory)

# Unit test class to test DETECTOR_REGISTRY
class TestDetectorRegistry(unittest.TestCase):
    
    def test_registry_contents(self):
        from base_miner.registry import Registry
        detector_registry = Registry()
        # Check if the registry has the expected keys (class names or custom names)
        registered_keys = list(detector_registry.data.keys())
        
        # Print all the registered models
        print("Registered detectors:")
        for name in registered_keys:
            print(f"Detector Name: {name}, Class: {detector_registry[name]}")
        
        # Assert that all expected keys are present
        self.assertEqual(len(registered_keys), 0, "There should be no registered detectors.")

    def test_registry_contents_after_import(self):
        from base_miner import DETECTOR_REGISTRY
        # Check if the registry has the expected keys (class names or custom names)
        registered_keys = list(DETECTOR_REGISTRY.data.keys())
        
        # Print all the registered models
        print("Registered detectors:")
        for name in registered_keys:
            print(f"Detector Name: {name}, Class: {DETECTOR_REGISTRY[name]}")
        
        # Assert that all expected keys are present
        self.assertIsNotNone(registered_keys, "Registered detectors should not be None")
        

if __name__ == '__main__':
    unittest.main()