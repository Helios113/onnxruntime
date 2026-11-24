"""Run all *_with_error op tests (MatMul / Gemm / ReduceSum) against the installed wheel in one go."""
import os
import sys
import unittest

import onnxruntime

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "onnxruntime", "test", "python", "contrib_ops")
sys.path.insert(0, TEST_DIR)
print("onnxruntime", onnxruntime.__version__, onnxruntime.__file__, onnxruntime.get_available_providers())
suite = unittest.defaultTestLoader.discover(TEST_DIR, pattern="test_*_with_error.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
