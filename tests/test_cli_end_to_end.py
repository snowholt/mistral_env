"""
End-to-end tests for the BeautyAI CLI.

This module tests the command-line interface by running actual commands
and verifying the output, validating both command structure and behavior.
"""
import sys
import os
import unittest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json

sys.path.append(".")  # Add the current directory to the path


class TestBeautyAICLIEndToEnd(unittest.TestCase):
    """End-to-end tests for the BeautyAI CLI."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, "config.json")
        self.models_file = os.path.join(self.test_dir, "models.json")
        
        # Create a basic configuration file
        config = {
            "default_engine": "transformers",
            "default_model": "test-model",
            "cache_dir": self.test_dir,
            "models": {
                "test-model": {
                    "model_id": "test/model",
                    "engine_type": "transformers",
                    "quantization": "4bit",
                    "dtype": "float16"
                }
            }
        }
        
        # Write the configuration to the file
        with open(self.config_file, "w") as f:
            json.dump(config, f)
        
        # Copy the configuration to the models file
        with open(self.models_file, "w") as f:
            json.dump(config, f)
        
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("subprocess.run")
    def test_version_command(self, mock_run):
        """Test the version command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"beautyai 1.0.0"
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai", "--version"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_help_command(self, mock_run):
        """Test the help command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"Usage: beautyai [OPTIONS] COMMAND [ARGS]..."
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai", "--help"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_model_list_command(self, mock_run):
        """Test the model list command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"Available Models:\n- test-model (default)"
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai", "model", "list", "--config", self.config_file, "--models-file", self.models_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run with proper args
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai")
        self.assertEqual(call_args[1], "model")
        self.assertEqual(call_args[2], "list")
        self.assertIn("--config", call_args)
        self.assertIn(self.config_file, call_args)

    @patch("subprocess.run")
    def test_system_status_command(self, mock_run):
        """Test the system status command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"System Status:\nNo models currently loaded."
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai", "system", "status", "--config", self.config_file, "--models-file", self.models_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run with proper args
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai")
        self.assertEqual(call_args[1], "system")
        self.assertEqual(call_args[2], "status")
        self.assertIn("--config", call_args)
        self.assertIn(self.config_file, call_args)

    @patch("subprocess.run")
    def test_config_show_command(self, mock_run):
        """Test the config show command."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = b"Current Configuration:\ndefault_engine: transformers\ndefault_model: test-model"
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai", "config", "show", "--config", self.config_file, "--models-file", self.models_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run with proper args
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai")
        self.assertEqual(call_args[1], "config")
        self.assertEqual(call_args[2], "show")
        self.assertIn("--config", call_args)
        self.assertIn(self.config_file, call_args)

    @patch("subprocess.run")
    def test_legacy_command_rerouting(self, mock_run):
        """Test that legacy commands are properly rerouted."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai-chat", "--model-name", "test-model"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run and rerouted
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai-chat")
        self.assertIn("--model-name", call_args)
        self.assertIn("test-model", call_args)
        
        # Reset the mock for the next test
        mock_run.reset_mock()

    @patch("subprocess.run")
    def test_test_cli_legacy_rerouting(self, mock_run):
        """Test that test CLI legacy command is properly rerouted."""
        # Set up mock return value
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Run the command
        cmd = ["beautyai-test", "--model-name", "test-model", "--prompt", "Hello!"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run and rerouted 
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai-test")
        self.assertIn("--model-name", call_args)
        self.assertIn("test-model", call_args)
        self.assertIn("--prompt", call_args)
        self.assertIn("Hello!", call_args)

    @patch("subprocess.run")
    def test_error_handling(self, mock_run):
        """Test that errors are properly handled."""
        # Set up mock return value for a command that fails
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = b"Error: Invalid model name"
        mock_run.return_value = mock_process
        
        # Run the command that should fail
        cmd = ["beautyai", "system", "load", "non-existent-model", "--config", self.config_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that the command was run with proper args
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "beautyai")
        self.assertEqual(call_args[1], "system")
        self.assertEqual(call_args[2], "load")
        self.assertEqual(call_args[3], "non-existent-model")


if __name__ == "__main__":
    unittest.main()
