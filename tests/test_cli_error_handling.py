"""
Error handling and edge case tests for the BeautyAI CLI.

This module focuses on testing error handling, edge cases, and input validation
to ensure the CLI behaves correctly when given invalid input or encountering errors.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import argparse
import logging

sys.path.append(".")  # Add the current directory to the path

from beautyai_inference.cli.unified_cli import UnifiedCLI


class TestBeautyAICLIErrorHandling(unittest.TestCase):
    """Tests for error handling in the BeautyAI CLI."""

    def setUp(self):
        """Set up the test environment."""
        # Create mocks for the services
        self.mock_model_registry_service = MagicMock()
        self.mock_lifecycle_service = MagicMock()
        self.mock_inference_service = MagicMock()
        self.mock_config_service = MagicMock()
        
        # Create the CLI with mocked services
        self.cli = UnifiedCLI()
        self.cli.model_registry_service = self.mock_model_registry_service
        self.mock_model_registry_service.list_models.return_value = 0
        self.cli.lifecycle_service = self.mock_lifecycle_service
        self.cli.inference_service = self.mock_inference_service
        self.cli.config_service = self.mock_config_service

    def test_invalid_command_group(self):
        """Test that invalid command groups are handled correctly."""
        args = argparse.Namespace()
        args.command_group = "invalid"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        with self.assertRaises(SystemExit):
            self.cli.route_command(args)

    def test_model_command_error_handling(self):
        """Test error handling in model commands."""
        # Mock the list_models method to raise an exception
        self.mock_model_registry_service.list_models.side_effect = ValueError("Test error")
        
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and print an error message
        with patch("sys.stderr"):
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)

    def test_system_command_error_handling(self):
        """Test error handling in system commands."""
        # Mock the show_status method to raise an exception
        self.mock_lifecycle_service.show_status.side_effect = RuntimeError("Test error")
        
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "status"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and print an error message
        with patch("sys.stderr"):
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)

    def test_run_command_error_handling(self):
        """Test error handling in run commands."""
        # Mock the start_chat method to raise an exception
        self.mock_inference_service.start_chat.side_effect = Exception("Test error")
        
        args = argparse.Namespace()
        args.command_group = "run"
        args.run_command = "chat"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and print an error message
        with patch("sys.stderr"):
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)

    def test_config_command_error_handling(self):
        """Test error handling in config commands."""
        # Mock the show_config method to raise an exception
        self.mock_config_service.show_config.side_effect = FileNotFoundError("Config file not found")
        
        args = argparse.Namespace()
        args.command_group = "config"
        args.config_command = "show"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and print an error message
        with patch("sys.stderr"):
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)

    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Test verbose mode
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = None
        args.models_file = None
        args.verbose = True
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        with patch("logging.Logger.setLevel") as mock_set_level:
            self.cli.route_command(args)
            mock_set_level.assert_called_with(logging.DEBUG)
        
        # Test quiet mode
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = True
        args.log_level = None
        args.no_color = False
        
        with patch("logging.Logger.setLevel") as mock_set_level:
            self.cli.route_command(args)
            mock_set_level.assert_called_with(logging.WARNING)
        
        # Test custom log level
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = "DEBUG"
        args.no_color = False
        
        with patch("logging.Logger.setLevel") as mock_set_level:
            self.cli.route_command(args)
            mock_set_level.assert_called_with(logging.DEBUG)

    def test_missing_required_args(self):
        """Test that missing required arguments are handled correctly."""
        # Mock the parser to check required arguments
        with patch.object(self.cli, "create_parser") as mock_create_parser:
            mock_parser = MagicMock()
            mock_parser.parse_args.side_effect = argparse.ArgumentError(
                None, "the following arguments are required"
            )
            mock_create_parser.return_value = mock_parser
            
            with self.assertRaises(SystemExit):
                with patch("sys.argv", ["beautyai", "model", "add"]):
                    from beautyai_inference.cli.unified_cli import main
                    main()

    def test_service_configuration_error(self):
        """Test error handling during service configuration."""
        # Mock the configure method to raise an exception
        self.mock_model_registry_service.configure.side_effect = ValueError("Invalid configuration")
        
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = "config.json"
        args.models_file = "models.json"
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and print an error message
        with patch("sys.stderr"):
            with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
                exit_code = self.cli.route_command(args)
                self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
