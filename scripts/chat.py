#!/usr/bin/env python3
"""
Interactive chat script for Mistral models.
This is a simple entry point that calls the chat_cli module.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mistral_inference.cli.chat_cli import main

if __name__ == "__main__":
    main()
