# Contributing Guide

Guidelines for contributing to the BeautyAI Inference Framework.

## 🤝 Welcome Contributors

Thank you for your interest in contributing to BeautyAI! This guide will help you get started with contributing to our AI inference framework.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment or discriminatory language
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing others' private information without permission

## 🚀 Getting Started

### Areas for Contribution

#### 🤖 Model Support
- Add new model architectures (Llama, Mistral, etc.)
- Improve quantization support
- Enhance model caching and lifecycle management
- Add new inference engines (vLLM, GGML, etc.)

#### 🎤 Voice Features
- Improve speech recognition accuracy
- Add new TTS voices and languages
- Enhance voice processing pipeline
- Optimize WebSocket performance

#### 🌐 API & Frontend
- Add new API endpoints
- Improve frontend UI/UX
- Enhance error handling
- Add new features and integrations

#### 📊 Performance & Monitoring
- Performance optimizations
- Monitoring and metrics
- Benchmarking tools
- Memory management improvements

#### 📚 Documentation
- API documentation
- User guides and tutorials
- Code examples
- Architecture documentation

## 💻 Development Setup

### Prerequisites
```bash
# System requirements
- Ubuntu 20.04+ or equivalent
- Python 3.11+
- Git
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 16GB+ RAM
```

### Clone and Setup
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/beautyai.git
cd beautyai

# 3. Set up upstream remote
git remote add upstream https://github.com/original-repo/beautyai.git

# 4. Create development environment
python -m venv venv
source venv/bin/activate

# 5. Install dependencies
cd backend
pip install -r requirements.txt
pip install -e .

cd ../frontend  
pip install -r requirements.txt

# 6. Install development tools
pip install pre-commit pytest black isort flake8 mypy
```

### Development Configuration
```bash
# Set up pre-commit hooks
pre-commit install

# Create development configuration
cp backend/src/model_registry.json.example backend/src/model_registry.json
cp frontend/config.json.example frontend/config.json

# Set environment variables
export BEAUTYAI_ENV=development
export BEAUTYAI_LOG_LEVEL=DEBUG
```

## 🔄 Contributing Process

### 1. Find or Create an Issue
```bash
# Check existing issues
# https://github.com/your-org/beautyai/issues

# Create new issue if needed
# Use issue templates for bug reports or feature requests
```

### 2. Create Feature Branch
```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### 3. Development Workflow
```bash
# Make your changes
# Run tests frequently
pytest

# Check code style
black .
isort .
flake8 .
mypy .

# Commit changes
git add .
git commit -m "feat: add new model support for Llama3"
```

### 4. Sync with Upstream
```bash
# Keep your branch up to date
git fetch upstream
git rebase upstream/main
```

## 📝 Code Standards

### Python Code Style

#### PEP 8 Compliance
```python
# Use Black for code formatting
black --line-length 88 .

# Use isort for import sorting  
isort .

# Follow PEP 8 naming conventions
class ModelManager:           # PascalCase for classes
    def load_model(self):     # snake_case for functions
        model_name = "..."    # snake_case for variables
        
# Use type hints
def process_text(text: str, model: Optional[Model] = None) -> str:
    """Process text with the given model."""
    pass
```

#### Docstring Standards
```python
def load_model(model_name: str, quantization: bool = False) -> Model:
    """Load a language model with optional quantization.
    
    Args:
        model_name: Name of the model to load from registry
        quantization: Whether to enable 4-bit quantization
        
    Returns:
        Loaded model instance ready for inference
        
    Raises:
        ModelNotFoundError: If model is not in registry
        InsufficientMemoryError: If not enough GPU memory
        
    Example:
        >>> model = load_model("qwen3-14b-instruct", quantization=True)
        >>> response = model.generate("Hello")
    """
    pass
```

### Project Structure Guidelines

#### New Features
```
backend/src/beautyai_inference/
├── services/
│   └── your_new_service/
│       ├── __init__.py
│       ├── service.py          # Main service logic
│       ├── models.py           # Data models
│       └── config.py           # Configuration
├── api/endpoints/
│   └── your_endpoint.py        # API endpoint
└── tests/
    └── test_your_service.py    # Unit tests
```

#### File Naming
- Use snake_case for Python files: `model_manager.py`
- Use PascalCase for classes: `ModelManager`
- Use descriptive names: `voice_service.py` not `voice.py`

### Git Commit Standards

#### Conventional Commits
```bash
# Format: type(scope): description

# Types:
feat: new feature
fix: bug fix
docs: documentation changes
style: formatting changes
refactor: code refactoring
test: adding tests
chore: maintenance tasks

# Examples:
feat(voice): add Arabic speech recognition support
fix(api): resolve memory leak in model loading
docs(readme): update installation instructions
test(voice): add unit tests for TTS service
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_model_manager.py
│   ├── test_voice_service.py
│   └── test_api_endpoints.py
├── integration/             # Integration tests
│   ├── test_model_loading.py
│   └── test_voice_pipeline.py
├── performance/             # Performance tests
│   └── test_benchmarks.py
└── fixtures/                # Test data and fixtures
    ├── sample_audio.wav
    └── test_models.json
```

### Writing Tests
```python
import pytest
from beautyai_inference.services.model import ModelManager

class TestModelManager:
    """Test cases for ModelManager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        return ModelManager()
    
    def test_load_model_success(self, model_manager):
        """Test successful model loading."""
        model = model_manager.load_model("test-model")
        assert model is not None
        assert model.is_loaded
    
    def test_load_nonexistent_model(self, model_manager):
        """Test loading non-existent model raises error."""
        with pytest.raises(ModelNotFoundError):
            model_manager.load_model("nonexistent-model")
    
    @pytest.mark.parametrize("quantization", [True, False])
    def test_quantization_options(self, model_manager, quantization):
        """Test model loading with different quantization settings."""
        model = model_manager.load_model("test-model", quantization=quantization)
        assert model.quantized == quantization
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_model_manager.py

# Run with coverage
pytest --cov=beautyai_inference --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
```

### Test Coverage Requirements
- Minimum 80% code coverage for new features
- 100% coverage for critical components (model loading, API endpoints)
- All public methods must have tests
- Edge cases and error conditions must be tested

## 📚 Documentation

### Code Documentation
```python
# Comprehensive docstrings for all public functions
def generate_text(
    self,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate text completion for the given prompt.
    
    This method uses the loaded language model to generate a text
    completion based on the input prompt and generation parameters.
    
    Args:
        prompt: Input text prompt for generation
        max_tokens: Maximum number of tokens to generate (default: 512)
        temperature: Sampling temperature between 0 and 1 (default: 0.7)
                    Higher values make output more random
    
    Returns:
        Generated text completion as a string
        
    Raises:
        ModelNotLoadedError: If no model is currently loaded
        InvalidParameterError: If parameters are outside valid ranges
        
    Example:
        >>> generator = TextGenerator()
        >>> response = generator.generate_text(
        ...     "What is artificial intelligence?",
        ...     max_tokens=256,
        ...     temperature=0.8
        ... )
        >>> print(response)
        "Artificial intelligence (AI) refers to..."
    """
```

### API Documentation
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    temperature: float = 0.7
    max_tokens: int = 512

@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> dict:
    """Chat with the AI model.
    
    Send a message to the AI model and receive a response.
    
    Args:
        request: Chat request containing message and parameters
        
    Returns:
        Dictionary containing the AI response
        
    Raises:
        HTTPException: 500 if model loading fails
        HTTPException: 400 if request parameters are invalid
    """
```

### User Documentation
- Update relevant documentation files in `docs/`
- Include usage examples
- Document configuration options
- Provide troubleshooting information

## 🔄 Pull Request Process

### Before Submitting
```bash
# 1. Ensure all tests pass
pytest

# 2. Check code style
black --check .
isort --check-only .
flake8 .
mypy .

# 3. Update documentation
# 4. Add/update tests for new features
# 5. Update CHANGELOG.md if applicable
```

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- List key changes
- Mention new files added
- Note any breaking changes

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] No breaking changes (or marked as breaking)
```

### Review Process
1. **Automated Checks**: CI/CD runs tests and code quality checks
2. **Code Review**: Maintainers review code for quality and design
3. **Testing**: Manual testing of new features
4. **Documentation**: Review of documentation updates
5. **Approval**: Two maintainer approvals required for merge

## 🏷️ Issue Labels

### Type Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to docs
- `performance`: Performance optimization
- `security`: Security-related issues

### Priority Labels
- `priority:high`: Critical issues
- `priority:medium`: Important but not urgent
- `priority:low`: Nice to have

### Component Labels
- `component:model`: Model loading and management
- `component:voice`: Voice processing features
- `component:api`: REST API endpoints
- `component:frontend`: Web UI components
- `component:tests`: Testing infrastructure

### Status Labels
- `status:needs-triage`: Needs initial review
- `status:in-progress`: Currently being worked on
- `status:blocked`: Blocked by external dependency
- `status:ready-for-review`: Ready for code review

## 🎯 Development Tips

### Performance Considerations
```python
# Use async/await for I/O operations
async def load_model_async(model_name: str):
    """Load model asynchronously to avoid blocking."""
    pass

# Implement proper caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_config(model_name: str):
    """Cache model configurations."""
    pass

# Use generators for large datasets
def process_large_dataset():
    """Use generators to process large datasets efficiently."""
    for item in large_dataset:
        yield process_item(item)
```

### Error Handling
```python
class BeautyAIError(Exception):
    """Base exception for BeautyAI errors."""
    pass

class ModelNotFoundError(BeautyAIError):
    """Raised when model is not found in registry."""
    pass

def load_model(model_name: str):
    """Load model with proper error handling."""
    try:
        # Model loading logic
        pass
    except FileNotFoundError:
        raise ModelNotFoundError(f"Model {model_name} not found")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise BeautyAIError(f"Failed to load model: {e}")
```

### Memory Management
```python
import torch
import gc

def cleanup_model(model):
    """Properly cleanup model from memory."""
    if hasattr(model, 'cpu'):
        model.cpu()
    del model
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
```

## 🆘 Getting Help

### Community Resources
- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Documentation**: Check existing documentation first

### Maintainer Contact
- Create an issue for bugs or feature requests
- Use discussions for questions and general help
- Tag maintainers (@maintainer) for urgent issues

### Development Support
- Join our development chat (link to be added)
- Attend community meetings (schedule to be announced)
- Review existing code for examples and patterns

---

**Thank you for contributing to BeautyAI!** 🚀

Your contributions help make AI more accessible and powerful for everyone.
