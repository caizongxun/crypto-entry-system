"""Crypto Entry System - Main Application Package"""

from flask import Flask
from flask_cors import CORS

__version__ = '1.0.0'
__author__ = 'Zong'


def create_app(config=None):
    """Application factory function."""
    app = Flask(__name__)
    
    # Configuration
    if config:
        app.config.update(config)
    
    # CORS
    CORS(app)
    
    return app
