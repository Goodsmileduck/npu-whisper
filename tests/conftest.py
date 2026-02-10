"""Shared fixtures for dictation engine tests."""
import sys
from pathlib import Path

# Add project root to path so we can import dictation_engine
sys.path.insert(0, str(Path(__file__).parent.parent))
