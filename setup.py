from setuptools import setup, find_packages

setup(
    name="beautyaiinference",
    version="1.0.0",
    description="A scalable, professional-grade CLI framework for running inference with Arabic AI models and multilingual language models",
    long_description="BeautyAI Inference Framework provides a unified CLI interface for managing and running inference with various language models, specializing in Arabic AI models but supporting multilingual capabilities. Features include modular architecture, multiple backend support (Transformers/vLLM), quantization capabilities, and comprehensive model lifecycle management.",
    author="Lumi AI",
    author_email="lumi@beautyai.dev",
    url="https://github.com/lumiai/beautyai-inference",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "psutil>=5.9.0",
        "argcomplete>=3.0.0",  # Add auto-completion support
        "torchaudio>=2.0.0",  # Audio processing
        "numpy>=1.21.0",  # Numerical operations
        "librosa>=0.9.0",  # Audio analysis
        "fastapi>=0.104.0",  # REST API framework
        "uvicorn[standard]>=0.24.0",  # ASGI server
        "pydantic>=2.0.0",  # Data validation
        "edge-tts>=6.1.0",  # Edge TTS for Python 3.11+
        "outetts>=0.1.0",  # OuteTTS for high-quality neural speech synthesis
        "llama-cpp-python>=0.2.0",  # LlamaCpp for GGUF models and OuteTTS
        "soundfile>=0.12.0",  # Audio file I/O for OuteTTS
        "huggingface-hub>=0.20.0",  # For downloading OuteTTS model
        "click>=8.0.0",  # CLI interface library
        "jsonschema>=4.0.0",  # JSON schema validation
        "python-multipart>=0.0.6",  # Form data handling for FastAPI
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "tts": [
            "soundfile>=0.12.0",  # Audio file I/O for OuteTTS
            "llama-cpp-python>=0.2.0",  # LlamaCpp for GGUF models
        ],
        "audio": ["soundfile>=0.12.0", "sox>=1.4.0"],  # Additional audio dependencies
        "dev": ["black", "isort", "flake8", "pytest"],
    },
    entry_points={
        "console_scripts": [
            # Primary unified CLI entry point
            "beautyai=beautyai_inference.cli.unified_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
