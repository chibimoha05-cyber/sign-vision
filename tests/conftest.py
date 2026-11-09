"""Pytest configuration and global test fixtures for SignVision."""

import os
import sys

# Ensure src/ directory is always in sys.path during pytest discovery and execution
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
