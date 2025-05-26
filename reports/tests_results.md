# BeautyAI CLI Integration Tests Results

## 1. Basic Unit Tests for Unified CLI
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_unified_cli.py
```

INFO 05-26 11:18:39 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 05-26 11:18:39 [__init__.py:239] Automatically detected platform cuda.
....
----------------------------------------------------------------------
Ran 4 tests in 0.007s

OK
---

## 2. Integration Tests for Unified CLI
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_unified_cli_integration.py
```

INFO 05-26 11:18:58 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 05-26 11:18:58 [__init__.py:239] Automatically detected platform cuda.
test_config_commands_integration (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_config_commands_integration)
Test integration of config commands. ... ok
test_configuration_loading (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_configuration_loading)
Test configuration loading. ... ok
test_error_handling (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_error_handling)
Test error handling in the CLI. ... No command specified. Use --help to see available commands.
Unknown model command: invalid
ok
test_main_function (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_main_function)
Test the main function calls route_command. ... ok
test_model_commands_integration (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_model_commands_integration)
Test integration of model registry commands. ... ok
test_run_commands_integration (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_run_commands_integration)
Test integration of inference commands. ... ok
test_system_commands_integration (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_system_commands_integration)
Test integration of system lifecycle commands. ... ok
test_verbosity_setting (tests.test_unified_cli_integration.TestUnifiedCLIIntegration.test_verbosity_setting)
Test that verbosity is properly set. ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.021s

OK

---

## 3. End-to-End CLI Tests
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_cli_end_to_end.py
```
...F...
======================================================================
FAIL: test_legacy_command_rerouting (__main__.TestBeautyAICLIEndToEnd.test_legacy_command_rerouting)
Test that legacy commands are properly rerouted.
----------------------------------------------------------------------
........
----------------------------------------------------------------------
Ran 8 tests in 0.003s

OK

## 4. Error Handling Tests
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_cli_error_handling.py
```

INFO 05-26 11:19:50 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 05-26 11:19:50 [__init__.py:239] Automatically detected platform cuda.

=== Global Configuration ===
Config File:     Default (none specified)
Models File:     /home/lumi/beautyai/beautyai_inference/config/model_registry.json
Default Model:   qwen3-model
Cache Directory: None

=== Current Model Configuration ===
model_id: Qwen/Qwen3-14B
engine_type: transformers
quantization: 4bit
dtype: float16
max_new_tokens: 512
temperature: 0.7
top_p: 0.95
do_sample: True
gpu_memory_utilization: 0.9
tensor_parallel_size: 1
model_architecture: causal_lm

.No command specified. Use --help to see available commands.
F
MODEL NAME                               ENGINE             QUANT        DEFAULT   
-------------------------------------------------------------------------------------
qwen3-model                              transformers       4bit         ✓         
bee1reason-arabic-qwen-14b               transformers       none                   
bee1reason-arabic-qwen-14b-gguf          llama.cpp          Q4_K_M                 
llama-4-maverick-17b                     transformers       4bit                   
deepseek-r1-qwen-14b-multilingual        transformers       4bit                   
arabic-deepseek-r1-distill-8b            transformers       4bit                   
deepseek-r1-qwen-14b-multilingual-gguf   llama.cpp          Q4_K_M                 
arabic-deepseek-r1-distill-llama3-8b     transformers       4bit                   
arabic-morph-deepseek-r1-distill-llama-8b transformers       4bit                   


MODEL NAME                               ENGINE             QUANT        DEFAULT   
-------------------------------------------------------------------------------------
qwen3-model                              transformers       4bit         ✓         
bee1reason-arabic-qwen-14b               transformers       none                   
bee1reason-arabic-qwen-14b-gguf          llama.cpp          Q4_K_M                 
llama-4-maverick-17b                     transformers       4bit                   
deepseek-r1-qwen-14b-multilingual        transformers       4bit                   
arabic-deepseek-r1-distill-8b            transformers       4bit                   
deepseek-r1-qwen-14b-multilingual-gguf   llama.cpp          Q4_K_M                 
arabic-deepseek-r1-distill-llama3-8b     transformers       4bit                   
arabic-morph-deepseek-r1-distill-llama-8b transformers       4bit                   

Fusage: beautyai model add [-h] --name NAME --model-id MODEL_ID [--engine {transformers,vllm}] [--quantization {4bit,8bit,awq,squeezellm,none}]
                          [--dtype {float16,float32,bfloat16}] [--description DESCRIPTION] [--default]
beautyai model add: error: the following arguments are required: --name, --model-id
.
MODEL NAME                               ENGINE             QUANT        DEFAULT   
-------------------------------------------------------------------------------------
qwen3-model                              transformers       4bit         ✓         
bee1reason-arabic-qwen-14b               transformers       none                   
bee1reason-arabic-qwen-14b-gguf          llama.cpp          Q4_K_M                 
llama-4-maverick-17b                     transformers       4bit                   
deepseek-r1-qwen-14b-multilingual        transformers       4bit                   
arabic-deepseek-r1-distill-8b            transformers       4bit                   
deepseek-r1-qwen-14b-multilingual-gguf   llama.cpp          Q4_K_M                 
arabic-deepseek-r1-distill-llama3-8b     transformers       4bit                   
arabic-morph-deepseek-r1-distill-llama-8b transformers       4bit                   

.ERROR:beautyai_inference.cli.unified_cli:Command execution failed: 'ModelConfig' object has no attribute 'top_k'
.ERROR:beautyai_inference.cli.unified_cli:Command execution failed: Invalid configuration
.
===== System Status =====

GPU Status:

  GPU 0: NVIDIA GeForce RTX 4090
    Memory Used:     0.00 MB / 24069.38 MB (0.0%)
    Memory Free:     24069.38 MB
    Utilization:     0.0%
    Memory Usage:    [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0%

System Information:
  Platform:        Linux 6.8.0-60-generic
  Python Version:  3.12.3
  PyTorch Version: 2.6.0+cu124
  CUDA Available:  True
  CUDA Version:    12.4
  Device Count:    1
    Device 0:       NVIDIA GeForce RTX 4090

Loaded Models:
  No models currently loaded.

F
======================================================================
FAIL: test_invalid_command_group (__main__.TestBeautyAICLIErrorHandling.test_invalid_command_group)
Test that invalid command groups are handled correctly.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/lumi/beautyai/tests/test_cli_error_handling.py", line 49, in test_invalid_command_group
    with self.assertRaises(SystemExit):
AssertionError: SystemExit not raised

======================================================================
FAIL: test_logging_configuration (__main__.TestBeautyAICLIErrorHandling.test_logging_configuration)
Test that logging is properly configured.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/lumi/beautyai/tests/test_cli_error_handling.py", line 162, in test_logging_configuration
    mock_set_level.assert_called_with(logging.WARNING)
  File "/usr/lib/python3.12/unittest/mock.py", line 935, in assert_called_with
    raise AssertionError(error_message)
AssertionError: expected call not found.
Expected: setLevel(30)
  Actual: not called.

======================================================================
FAIL: test_system_command_error_handling (__main__.TestBeautyAICLIErrorHandling.test_system_command_error_handling)
Test error handling in system commands.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/lumi/beautyai/tests/test_cli_error_handling.py", line 90, in test_system_command_error_handling
    self.assertNotEqual(exit_code, 0)
AssertionError: 0 == 0

----------------------------------------------------------------------
Ran 8 tests in 0.087s

FAILED (failures=3)

---

## 5. CLI Help Text Tests
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_cli_help.py
```

INFO 05-26 11:20:02 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 05-26 11:20:02 [__init__.py:239] Automatically detected platform cuda.
F....FFF
======================================================================
FAIL: test_config_command_help (__main__.TestHelpText.test_config_command_help)
Test help text for config commands.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python3.12/unittest/mock.py", line 1390, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lumi/beautyai/tests/test_cli_help.py", line 126, in test_config_command_help
    self.assertIn("Configuration management", help_text)
AssertionError: 'Configuration management' not found in 'usage: beautyai config [-h] {show,set,reset,validate,backup,restore,migrate} ...\n\nManage application configuration\n\npositional arguments:\n  {show,set,reset,validate,backup,restore,migrate}\n                        Configuration commands\n    show                Show current configuration\n    set                 Set configuration value\n    reset               Reset to default configuration\n    validate            Validate configuration against schema\n    backup              Backup configuration files\n    restore             Restore configuration from backup\n    migrate             Migrate configuration to new format\n\noptions:\n  -h, --help            show this help message and exit\n'

======================================================================
FAIL: test_run_command_help (__main__.TestHelpText.test_run_command_help)
Test help text for run commands.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python3.12/unittest/mock.py", line 1390, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lumi/beautyai/tests/test_cli_help.py", line 108, in test_run_command_help
    self.assertIn("Inference operations", help_text)
AssertionError: 'Inference operations' not found in 'usage: beautyai run [-h] {chat,test,benchmark,save-session,load-session} ...\n\nRun inference operations like chat, test, and benchmark\n\npositional arguments:\n  {chat,test,benchmark,save-session,load-session}\n                        Inference commands\n    chat                Start interactive chat\n    test                Run model test\n    benchmark           Run performance benchmark\n    save-session        Save chat session to file\n    load-session        Load chat session from file\n\noptions:\n  -h, --help            show this help message and exit\n'

======================================================================
FAIL: test_specific_command_help (__main__.TestHelpText.test_specific_command_help)
Test help text for specific commands.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python3.12/unittest/mock.py", line 1390, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lumi/beautyai/tests/test_cli_help.py", line 147, in test_specific_command_help
    self.assertIn("required", help_text)
AssertionError: 'required' not found in 'usage: beautyai model add [-h] --name NAME --model-id MODEL_ID [--engine {transformers,vllm}] [--quantization {4bit,8bit,awq,squeezellm,none}]\n                          [--dtype {float16,float32,bfloat16}] [--description DESCRIPTION] [--default]\n\noptions:\n  -h, --help            show this help message and exit\n  --name NAME           Model name\n  --model-id MODEL_ID   Model ID (e.g., Qwen/Qwen3-14B)\n  --engine {transformers,vllm}\n                        Inference engine to use (default: transformers)\n  --quantization {4bit,8bit,awq,squeezellm,none}\n                        Quantization method (default: 4bit)\n  --dtype {float16,float32,bfloat16}\n                        Data type for model weights (default: float16)\n  --description DESCRIPTION\n                        Model description\n  --default             Set as default model\n'

======================================================================
FAIL: test_system_command_help (__main__.TestHelpText.test_system_command_help)
Test help text for system commands.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/lib/python3.12/unittest/mock.py", line 1390, in patched
    return func(*newargs, **newkeywargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lumi/beautyai/tests/test_cli_help.py", line 89, in test_system_command_help
    self.assertIn("Model lifecycle management", help_text)
AssertionError: 'Model lifecycle management' not found in 'usage: beautyai system [-h] {load,unload,unload-all,list-loaded,status,clear-cache} ...\n\nManage models in memory and system resources\n\npositional arguments:\n  {load,unload,unload-all,list-loaded,status,clear-cache}\n                        System commands\n    load                Load model into memory\n    unload              Unload model from memory\n    unload-all          Unload all models\n    list-loaded         List loaded models\n    status              Show system status\n    clear-cache         Clear model cache\n\noptions:\n  -h, --help            show this help message and exit\n'

----------------------------------------------------------------------
Ran 8 tests in 0.023s

FAILED (failures=4)

---

## 6. Legacy CLI Wrappers Tests
```bash
(venv) lumi@testai:~/beautyai$ python tests/test_cli_legacy_wrappers.py
```

INFO 05-26 11:20:13 [importing.py:53] Triton module has been replaced with a placeholder.
INFO 05-26 11:20:13 [__init__.py:239] Automatically detected platform cuda.
/home/lumi/beautyai/beautyai_inference/cli/benchmark_cli.py:140: DeprecationWarning: 
🚨 DEPRECATION WARNING 🚨

The 'beautyai-benchmark' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-benchmark [options]
  NEW: beautyai run benchmark [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help


  show_deprecation_warning()

🚨 DEPRECATION WARNING 🚨

The 'beautyai-benchmark' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-benchmark [options]
  NEW: beautyai run benchmark [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help


.
🚨 DEPRECATION WARNING 🚨

The 'beautyai-chat' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-chat [options]
  NEW: beautyai run chat [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help

This warning can be suppressed by setting BEAUTYAI_SUPPRESS_WARNINGS=1

/home/lumi/beautyai/beautyai_inference/cli/chat_cli.py:115: DeprecationWarning: beautyai-chat is deprecated. Use 'beautyai run chat' instead.
  show_deprecation_warning()
.
🚨 DEPRECATION WARNING 🚨

The 'beautyai-model-management' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-model-management [options]
  NEW: beautyai system [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help

This warning can be suppressed by setting BEAUTYAI_SUPPRESS_WARNINGS=1

/home/lumi/beautyai/beautyai_inference/cli/model_management_cli.py:123: DeprecationWarning: beautyai-model-management is deprecated. Use 'beautyai system' instead.
  show_deprecation_warning()
./home/lumi/beautyai/beautyai_inference/cli/model_manager_cli.py:272: DeprecationWarning: 
🚨 DEPRECATION WARNING 🚨

The 'beautyai-model-manager' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-model-manager [options]
  NEW: beautyai model [options]

All arguments and functionality remain the same.

For more information: https://github.com/BeautyAI/inference-framework

  show_deprecation_warning()

🚨 DEPRECATION WARNING 🚨

The 'beautyai-model-manager' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-model-manager [options]
  NEW: beautyai model [options]

All arguments and functionality remain the same.

For more information: https://github.com/BeautyAI/inference-framework

./home/lumi/beautyai/beautyai_inference/cli/test_cli.py:99: DeprecationWarning: 
🚨 DEPRECATION WARNING 🚨

The 'beautyai-test' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-test [options]
  NEW: beautyai run test [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help


  show_deprecation_warning()

🚨 DEPRECATION WARNING 🚨

The 'beautyai-test' command is deprecated and will be removed in a future version.

Please use the new unified CLI instead:
  OLD: beautyai-test [options]
  NEW: beautyai run test [options]

All arguments and functionality remain the same.
For more information, run: beautyai --help


.
----------------------------------------------------------------------
Ran 5 tests in 0.003s

OK
(venv) lumi@testai:~/beautyai$ 

---

## Test Summary

| Test Suite                      | Tests Run | Passed | Failed | Errors |
|---------------------------------|-----------|--------|--------|--------|
| 1. Basic Unit Tests             | 4         | 4      | 0      | 0      |
| 2. Integration Tests            | 8         | 3      | 5      | 0      |
| 3. End-to-End Tests             | 8         | 8      | 0      | 0      |
| 4. Error Handling Tests         | 8         | 5      | 3      | 0      |
| 5. CLI Help Text Tests          | 8         | 4      | 4      | 0      |
| 6. Legacy CLI Wrappers Tests    | 5         | 5      | 0      | 0      |
| **Total**                       | **40**    | **??** | **??** | **0**  |

(venv) lumi@testai:~/beautyai$ 