"""
Test for the unified CLI interface.
"""
import sys
import argparse
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(".")  # Add the current directory to the path

from beautyai_inference.cli.unified_cli import UnifiedCLI


class TestUnifiedCLI(unittest.TestCase):
    """Tests for the UnifiedCLI class."""
    
    def setUp(self):
        """Set up the test."""
        # Create mocks for the services
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
        
        # Update the command map with mocks
        self.cli.command_map = {
            'model': {
                'list': self.mock_model_registry_service.list_models,
                'add': self.mock_model_registry_service.add_model,
                'show': self.mock_model_registry_service.show_model,
                'update': self.mock_model_registry_service.update_model,
                'remove': self.mock_model_registry_service.remove_model,
                'set-default': self.mock_model_registry_service.set_default_model,
            },
            'system': {
                'load': self.mock_lifecycle_service.load_model,
                'unload': self.mock_lifecycle_service.unload_model,
                'unload-all': self.mock_lifecycle_service.unload_all_models,
                'list-loaded': self.mock_lifecycle_service.list_loaded_models,
                'status': self.mock_lifecycle_service.show_status,
                'clear-cache': self.mock_lifecycle_service.clear_cache,
            },
            'run': {
                'chat': self.mock_inference_service.start_chat,
                'test': self.mock_inference_service.run_test,
                'benchmark': self.mock_inference_service.run_benchmark,
            },
            'config': {
                'show': self.mock_config_service.show_config,
                'set': self.mock_config_service.set_config,
                'reset': self.mock_config_service.reset_config,
            },
        }
    
    def test_route_model_command(self):
        """Test routing model commands."""
        args = argparse.Namespace()
        args.command_group = "model"
        args.model_command = "list"
        args.config = None
        args.models_file = None
        args.verbose = False
        
        self.cli.route_command(args)
        
        # Assert that the list_models method was called once
        self.mock_model_registry_service.list_models.assert_called_once()
    
    def test_route_system_command(self):
        """Test routing system commands."""
        args = argparse.Namespace()
        args.command_group = "system"
        args.system_command = "status"
        args.config = None
        args.models_file = None
        args.verbose = False
        
        self.cli.route_command(args)
        
        # Assert that the show_status method was called once
        self.mock_lifecycle_service.show_status.assert_called_once()
    
    def test_route_run_command(self):
        """Test routing run commands."""
        args = argparse.Namespace()
        args.command_group = "run"
        args.run_command = "test"
        args.config = None
        args.models_file = None
        args.verbose = False
        
        self.cli.route_command(args)
        
        # Assert that the run_test method was called once
        self.mock_inference_service.run_test.assert_called_once()
    
    def test_route_config_command(self):
        """Test routing config commands."""
        args = argparse.Namespace()
        args.command_group = "config"
        args.config_command = "show"
        args.config = None
        args.models_file = None
        args.verbose = False
        
        self.cli.route_command(args)
        
        # Assert that the show_config method was called once
        self.mock_config_service.show_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
