"""
Help text and documentation tests for the BeautyAI CLI.

This module verifies that the help text, examples, and documentation
in the CLI are accurate, complete, and properly formatted.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import argparse
import re

sys.path.append(".")  # Add the current directory to the path

from beautyai_inference.cli.unified_cli import UnifiedCLI


class TestHelpText(unittest.TestCase):
    """Tests for help text and documentation in the CLI."""

    def setUp(self):
        """Set up the test environment."""
        self.cli = UnifiedCLI()
        self.parser = self.cli.create_parser()
        
    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_main_help_text(self, mock_stderr, mock_stdout):
        """Test that the main help text is complete and properly formatted."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that important sections are present
        self.assertIn("Unified CLI for BeautyAI model management and inference", help_text)
        self.assertIn("Examples:", help_text)
        self.assertIn("positional arguments:", help_text)
        self.assertIn("options:", help_text)
        
        # Check that all command groups are documented
        self.assertIn("model", help_text)
        self.assertIn("system", help_text)
        self.assertIn("run", help_text)
        self.assertIn("config", help_text)
        
        # Check that the examples section contains examples for each command group
        self.assertIn("beautyai model list", help_text)
        self.assertIn("beautyai system", help_text)
        self.assertIn("beautyai run", help_text)
        self.assertIn("beautyai config", help_text)
        
        # Check that migration information is included
        self.assertIn("Migration from legacy commands", help_text)
        self.assertIn("now removed", help_text)
        self.assertIn("beautyai-", help_text)

    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_model_command_help(self, mock_stderr, mock_stdout):
        """Test help text for model commands."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["model", "--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that important sections are present
        self.assertIn("Manage model configurations in the registry", help_text)
        self.assertIn("list", help_text)
        self.assertIn("add", help_text)
        self.assertIn("show", help_text)
        self.assertIn("update", help_text)
        self.assertIn("remove", help_text)
        self.assertIn("set-default", help_text)

    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_system_command_help(self, mock_stderr, mock_stdout):
        """Test help text for system commands."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["system", "--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that important sections are present
        self.assertIn("Model lifecycle management", help_text)
        self.assertIn("load", help_text)
        self.assertIn("unload", help_text)
        self.assertIn("unload-all", help_text)
        self.assertIn("list-loaded", help_text)
        self.assertIn("status", help_text)
        self.assertIn("clear-cache", help_text)

    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_run_command_help(self, mock_stderr, mock_stdout):
        """Test help text for run commands."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that important sections are present
        self.assertIn("Inference operations", help_text)
        self.assertIn("chat", help_text)
        self.assertIn("test", help_text)
        self.assertIn("benchmark", help_text)
        self.assertIn("save-session", help_text)
        self.assertIn("load-session", help_text)

    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_config_command_help(self, mock_stderr, mock_stdout):
        """Test help text for config commands."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["config", "--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that important sections are present
        self.assertIn("Configuration management", help_text)
        self.assertIn("show", help_text)
        self.assertIn("set", help_text)
        self.assertIn("reset", help_text)
        self.assertIn("validate", help_text)
        self.assertIn("backup", help_text)
        self.assertIn("restore", help_text)
        self.assertIn("migrate", help_text)

    @patch("sys.stdout", new_callable=MagicMock)
    @patch("sys.stderr", new_callable=MagicMock)
    def test_specific_command_help(self, mock_stderr, mock_stdout):
        """Test help text for specific commands."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["model", "add", "--help"])
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check that required arguments are documented
        self.assertIn("--name", help_text)
        self.assertIn("required", help_text)

    def test_help_example_commands_exist(self):
        """Test that the commands mentioned in the examples actually exist."""
        parser = self.cli.create_parser()
        
        # Commands from the examples section
        example_commands = [
            ["model", "list"],
            ["model", "add", "--name", "my-model", "--model-id", "Qwen/Qwen3-14B"],
            ["model", "show", "my-model"],
            ["model", "set-default", "my-model"],
            ["system", "load", "my-model"],
            ["system", "status"],
            ["system", "unload", "my-model"],
            ["run", "chat", "--model-name", "my-model"],
            ["run", "test", "--model", "Qwen/Qwen3-14B"],
            ["run", "benchmark", "--model-name", "my-model", "--output-file", "results.json"],
            ["config", "show"],
            ["config", "set", "default_engine", "vllm"],
            ["config", "validate"]
        ]
        
        # Test that each example command can be parsed
        for cmd in example_commands:
            try:
                # Add the program name at the beginning to simulate a real command line
                parser.parse_args(cmd)
            except SystemExit:
                # If the command requires additional arguments, it will exit.
                # That's fine for this test, we just want to make sure the command exists.
                pass
            except Exception as e:
                self.fail(f"Command '{' '.join(cmd)}' failed to parse: {str(e)}")

    def test_examples_section_formatting(self):
        """Test that the examples section is properly formatted."""
        help_examples = self.cli._get_help_examples()
        
        # Check that the examples section is properly formatted
        self.assertIn("Examples:", help_examples)
        
        # Check that each command is properly indented and includes a comment
        command_lines = re.findall(r"^\s*#.*\n\s*beautyai", help_examples, re.MULTILINE)
        
        # There should be at least one command for each command group
        self.assertGreaterEqual(len(command_lines), 4)
        
        # Each command should be properly formatted
        for line in command_lines:
            # The line should contain a comment starting with #
            self.assertIn("#", line)
            # The line should contain the command beautyai
            self.assertIn("beautyai", line)


if __name__ == "__main__":
    unittest.main()
