"""
Tests for backward compatibility wrappers in BeautyAI CLI.

This module tests the legacy CLI wrappers to ensure they properly redirect
to the unified CLI interface.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

sys.path.append(".")  # Add the current directory to the path

from beautyai_inference.cli.chat_cli import main as chat_main
from beautyai_inference.cli.test_cli import main as test_main
from beautyai_inference.cli.benchmark_cli import main as benchmark_main
from beautyai_inference.cli.model_manager_cli import main as model_manager_main
from beautyai_inference.cli.model_management_cli import main as model_management_main


class TestLegacyCLIWrappers(unittest.TestCase):
    """Tests for legacy CLI wrappers."""

    def setUp(self):
        """Set up the test environment."""
        self.original_argv = sys.argv
        
    def tearDown(self):
        """Tear down the test environment."""
        sys.argv = self.original_argv

    @patch("beautyai_inference.cli.chat_cli.log_legacy_usage")
    @patch("subprocess.run")
    def test_chat_cli_redirection(self, mock_run, mock_log_usage):
        """Test that chat CLI redirects to unified CLI."""
        # Set up
        sys.argv = ["beautyai-chat", "--model-name", "test-model"]
        
        # Call the main function with exit suppressed
        with self.assertRaises(SystemExit):
            chat_main()
        
        # Assert
        mock_log_usage.assert_called_once_with("beautyai-chat", ["--model-name", "test-model"])
        mock_run.assert_called_once()
        # Get the run call arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # Check that it redirects to the unified CLI
        self.assertEqual(cmd[0], "beautyai")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "chat")
        self.assertIn("--model-name", cmd)
        self.assertIn("test-model", cmd)

    @patch("beautyai_inference.cli.test_cli.log_legacy_usage")
    @patch("subprocess.run")
    def test_test_cli_redirection(self, mock_run, mock_log_usage):
        """Test that test CLI redirects to unified CLI."""
        # Set up
        sys.argv = ["beautyai-test", "--model-name", "test-model", "--prompt", "Hello"]
        
        # Call the main function with exit suppressed
        with self.assertRaises(SystemExit):
            test_main()
        
        # Assert
        mock_log_usage.assert_called_once_with("beautyai-test", ["--model-name", "test-model", "--prompt", "Hello"])
        mock_run.assert_called_once()
        # Get the run call arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # Check that it redirects to the unified CLI
        self.assertEqual(cmd[0], "beautyai")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "test")
        self.assertIn("--model-name", cmd)
        self.assertIn("test-model", cmd)
        self.assertIn("--prompt", cmd)
        self.assertIn("Hello", cmd)

    @patch("beautyai_inference.cli.benchmark_cli.log_legacy_usage")
    @patch("subprocess.run")
    def test_benchmark_cli_redirection(self, mock_run, mock_log_usage):
        """Test that benchmark CLI redirects to unified CLI."""
        # Set up
        sys.argv = ["beautyai-benchmark", "--model-name", "test-model", "--iterations", "5"]
        
        # Call the main function with exit suppressed
        with self.assertRaises(SystemExit):
            benchmark_main()
        
        # Assert
        mock_log_usage.assert_called_once_with("beautyai-benchmark", ["--model-name", "test-model", "--iterations", "5"])
        mock_run.assert_called_once()
        # Get the run call arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # Check that it redirects to the unified CLI
        self.assertEqual(cmd[0], "beautyai")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "benchmark")
        self.assertIn("--model-name", cmd)
        self.assertIn("test-model", cmd)
        self.assertIn("--iterations", cmd)
        self.assertIn("5", cmd)

    @patch("beautyai_inference.cli.model_manager_cli.log_legacy_usage")
    @patch("subprocess.run")
    def test_model_manager_cli_redirection(self, mock_run, mock_log_usage):
        """Test that model manager CLI redirects to unified CLI."""
        # Set up
        sys.argv = ["beautyai-model-manager", "list"]
        
        # Call the main function with exit suppressed
        with self.assertRaises(SystemExit):
            model_manager_main()
        
        # Assert
        mock_log_usage.assert_called_once_with("beautyai-model-manager", ["list"])
        mock_run.assert_called_once()
        # Get the run call arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # Check that it redirects to the unified CLI
        self.assertEqual(cmd[0], "beautyai")
        self.assertEqual(cmd[1], "model")
        self.assertEqual(cmd[2], "list")

    @patch("beautyai_inference.cli.model_management_cli.log_legacy_usage")
    @patch("subprocess.run")
    def test_model_management_cli_redirection(self, mock_run, mock_log_usage):
        """Test that model management CLI redirects to unified CLI."""
        # Set up
        sys.argv = ["beautyai-model-management", "load", "test-model"]
        
        # Call the main function with exit suppressed
        with self.assertRaises(SystemExit):
            model_management_main()
        
        # Assert
        mock_log_usage.assert_called_once_with("beautyai-model-management", ["load", "test-model"])
        mock_run.assert_called_once()
        # Get the run call arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # Check that it redirects to the unified CLI
        self.assertEqual(cmd[0], "beautyai")
        self.assertEqual(cmd[1], "system")
        self.assertEqual(cmd[2], "load")
        self.assertIn("test-model", cmd)


if __name__ == "__main__":
    unittest.main()
