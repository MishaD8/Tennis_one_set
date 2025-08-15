#!/usr/bin/env python3
"""
Pytest configuration file for Tennis One Set tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

@pytest.fixture(scope="session")
def app():
    """Create and configure a test app instance."""
    from src.api.app import create_app
    from src.config.config import TestingConfig
    
    app = create_app()
    app.config.from_object(TestingConfig)
    
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()