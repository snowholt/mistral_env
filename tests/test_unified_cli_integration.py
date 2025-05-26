"""
Integration tests for the unified CLI interface of BeautyAI.

This module provides comprehensive tests for the unified CLI interface,
including command routing, error handling, and integration between different
components.
"""
import sys
import os
import json
import argparse
import unittest
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil

sys.path.append(".")  # Add the current directory to the path

from beautyai_inference.cli.unified_cli import UnifiedCLI
from beautyai_inference.config.config_manager import ModelConfig, AppConfig


class TestUnifiedCLIIntegration(unittest.TestCase):
    """Integration tests for the UnifiedCLI class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "config.json")
        self.models_file = os.path.join(self.test_dir, "models.json")
        
        # Create mock services
        self.mock_model_registry_service = MagicMock()
        self.mock_lifecycle_service = MagicMock()
        self.mock_inference_service = MagicMock()
        self.mock_config_service = MagicMock()
        
        # Create the CLI with mocked services
        self.cli = UnifiedCLI()
        self.cli.model_registry_service = self.mock_model_registry_service
        self.cli.lifecycle_service = self.mock_lifecycle_service
        self.cli.inference_service = self.mock_inference_service
        self.cli.config_service = self.mock_config_service
        
        # Create a sample app config and model config for testing
        self.app_config = AppConfig()
        self.model_config = ModelConfig(model_id="test/model")
        
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("sys.argv", ["beautyai", "model", "list"])
    @patch("beautyai_inference.cli.unified_cli.UnifiedCLI.route_command")
    def test_main_function(self, mock_route):
        """Test the main function calls route_command."""
        with patch.object(self.cli, "create_parser", return_value=MagicMock()):
            with patch("beautyai_inference.cli.unified_cli.UnifiedCLI", return_value=self.cli):
                from beautyai_inference.cli.unified_cli import main
                main()
                mock_route.assert_called_once()
    
    def test_model_commands_integration(self):
        """Test integration of model registry commands."""
        # Test list models
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_model_registry_service.list_models.assert_called_once()
        
        # Test add model
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "add"
        args.name = "test-model"
        args.model_id = "test/model"
        args.engine = "transformers"
        args.quantization = "4bit"
        args.description = "Test model"
        args.default = False
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_model_registry_service.add_model.assert_called_once()
        
        # Test update model
        self.mock_model_registry_service.add_model.reset_mock()
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "update"
        args.name = "test-model"
        args.model_id = "updated/model"
        args.engine = "vllm"
        args.quantization = "8bit"
        args.description = "Updated model"
        args.default = True
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_model_registry_service.update_model.assert_called_once()
        
        # Test remove model
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "remove"
        args.name = "test-model"
        args.clear_cache = True
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_model_registry_service.remove_model.assert_called_once()

    def test_system_commands_integration(self):
        """Test integration of system lifecycle commands."""
        # Test load model
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "load"
        args.model_name = "test-model"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_lifecycle_service.load_model.assert_called_once()
        
        # Test status
        self.mock_lifecycle_service.load_model.reset_mock()
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "status"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_lifecycle_service.show_status.assert_called_once()
        
        # Test unload model
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "unload"
        args.model_name = "test-model"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_lifecycle_service.unload_model.assert_called_once()

    def test_run_commands_integration(self):
        """Test integration of inference commands."""
        # Test chat
        args = argparse.Namespace()
        args.command_group = "run"
        args.run_command = "chat"
        args.model_name = "test-model"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_inference_service.start_chat.assert_called_once()
        
        # Test test
        self.mock_inference_service.start_chat.reset_mock()
        args = argparse.Namespace()
        args.command_group = "run"
        args.run_command = "test"
        args.model_name = "test-model"
        args.prompt = "Hello, model!"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_inference_service.run_test.assert_called_once()
        
        # Test benchmark
        args = argparse.Namespace()
        args.command_group = "run"
        args.run_command = "benchmark"
        args.model_name = "test-model"
        args.iterations = 5
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_inference_service.run_benchmark.assert_called_once()

    def test_config_commands_integration(self):
        """Test integration of config commands."""
        # Test show config
        args = argparse.Namespace()
        args.command_group = "config"
        args.config_command = "show"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_config_service.show_config.assert_called_once()
        
        # Test set config
        self.mock_config_service.show_config.reset_mock()
        args = argparse.Namespace()
        args.command_group = "config"
        args.config_command = "set"
        args.key = "default_engine"
        args.value = "vllm"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_config_service.set_config.assert_called_once()
        
        # Test validate config
        args = argparse.Namespace()
        args.command_group = "config"
        args.config_command = "validate"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        self.cli.route_command(args)
        self.mock_config_service.validate_config.assert_called_once()

    def test_error_handling(self):
        """Test error handling in the CLI."""
        # Test invalid command group
        args = argparse.Namespace()
        args.command_group = "invalid"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        with self.assertRaises(SystemExit):
            self.cli.route_command(args)
        
        # Test invalid model command
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "invalid"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        
        with self.assertRaises(SystemExit):
            self.cli.route_command(args)
    
    @patch("logging.Logger.setLevel")
    def test_verbosity_setting(self, mock_set_level):
        """Test that verbosity is properly set."""
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = True
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        self.cli.route_command(args)
        mock_set_level.assert_called_with(logging.DEBUG)

    @patch("json.load")
    @patch("builtins.open", new_callable=mock_open)
    def test_configuration_loading(self, mock_file_open, mock_json_load):
        """Test configuration loading."""
        # Setup mock config
        mock_json_load.return_value = {
            "default_engine": "transformers",
            "models": {
                "test-model": {
                    "model_id": "test/model",
                    "engine_type": "transformers",
                    "quantization": "4bit"
                }
            }
        }
        
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = self.config_file
        args.models_file = self.models_file
        args.verbose = False
        args.quiet = False
        args.log_level = None
        args.no_color = False
        
        # Configure services
        self.cli.route_command(args)
        
        # Assert configuration was loaded (mock_file_open should be called)
        mock_file_open.assert_called()
        mock_json_load.assert_called()
        
        # Assert that the configuration was passed to all services
        self.mock_model_registry_service.configure.assert_called_once()
        self.mock_lifecycle_service.configure.assert_called_once()
        self.mock_inference_service.configure.assert_called_once()
        self.mock_config_service.configure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
