# conftest.py — project-level pytest configuration
# Ensures the workspace root is on sys.path for all test invocations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
