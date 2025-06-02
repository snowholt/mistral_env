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
        # Create a mock adapter to inject mocked services
        self.mock_adapter = MagicMock()
        self.mock_config_service = MagicMock()
        
        # Set default return values to avoid None returns
        self.mock_adapter.list_models.return_value = 0
        self.mock_adapter.show_status.return_value = 0
        self.mock_adapter.start_chat.return_value = 0
        self.mock_config_service.show_config.return_value = 0
        
        # Create the CLI with mocked adapter and config service
        self.cli = UnifiedCLI(
            adapter=self.mock_adapter,
            config_service=self.mock_config_service
        )

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
        
        with patch('sys.exit') as mock_exit:
            self.cli.route_command(args)
            mock_exit.assert_called_with(1)

    def test_model_command_error_handling(self):
        """Test error handling in model commands."""
        # Mock the list_models method to raise an exception
        self.mock_adapter.list_models.side_effect = ValueError("Test error")
        
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
        self.mock_adapter.show_status.side_effect = RuntimeError("Test error")
        
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "status"
        args.config = None
        args.models_file = None
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # The CLI should catch the exception and return an error code
        with patch("sys.stderr"):  # Suppress error output during test
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)  # Should not return success

    def test_run_command_error_handling(self):
        """Test error handling in run commands."""
        # Mock the start_chat method to raise an exception
        self.mock_adapter.start_chat.side_effect = Exception("Test error")
        
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
        
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            self.cli.route_command(args)
            mock_logger.setLevel.assert_called_with(logging.DEBUG)
        
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
        
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            self.cli.route_command(args)
            mock_logger.setLevel.assert_called_with(logging.WARNING)
        
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
        
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            self.cli.route_command(args)
            mock_logger.setLevel.assert_called_with(logging.DEBUG)

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
        # Mock the list_models method to raise an exception due to incomplete configuration
        self.mock_adapter.list_models.side_effect = ValueError("Configuration missing")
        
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
            exit_code = self.cli.route_command(args)
            self.assertNotEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
